// Copyright 2026 The Dawn & Tint Authors
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice, this
//    list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
//
// 3. Neither the name of the copyright holder nor the names of its
//    contributors may be used to endorse or promote products derived from
//    this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#include "src/dawn/native/metal/ResourceTableMTL.h"

#include <algorithm>
#include <utility>
#include <vector>

#include "src/dawn/common/Enumerator.h"
#include "src/dawn/common/MatchVariant.h"
#include "src/dawn/native/DynamicUploader.h"
#include "src/dawn/native/ResourceTableDefaultResources.h"
#include "src/dawn/native/metal/BufferMTL.h"
#include "src/dawn/native/metal/CommandRecordingContext.h"
#include "src/dawn/native/metal/DeviceMTL.h"
#include "src/dawn/native/metal/QueueMTL.h"
#include "src/dawn/native/metal/SamplerMTL.h"
#include "src/dawn/native/metal/TextureMTL.h"
#include "src/dawn/native/metal/UtilsMetal.h"
#include "src/utils/span.h"

namespace dawn::native::metal {

// static
ResultOrError<Ref<ResourceTable>> ResourceTable::Create(Device* device,
                                                        const ResourceTableDescriptor* descriptor) {
    Ref<ResourceTable> table = AcquireRef(new ResourceTable(device, descriptor));
    DAWN_TRY(table->Initialize());
    return table;
}

ResourceTable::~ResourceTable() = default;

MaybeError ResourceTable::Initialize() {
    DAWN_TRY(ResourceTableBase::InitializeBase());

    Device* device = ToBackend(GetDevice());
    id<MTLDevice> mtlDevice = device->GetMTLDevice();

    uint32_t totalSlots = uint32_t{GetSizeWithDefaultResources()};
    NSUInteger tableByteLength = totalSlots * sizeof(uint64_t);
    if (tableByteLength == 0) {
        tableByteLength = sizeof(uint64_t);
    }

    mTableBuffer = AcquireNSPRef([mtlDevice newBufferWithLength:tableByteLength
                                                        options:MTLResourceStorageModeShared]);
    memset([*mTableBuffer contents], 0, tableByteLength);

    // Initialize default resource slots [mAPISize .. mAPISize + 34]
    ResourceTableDefaultResources* defaultResources;
    DAWN_TRY_ASSIGN(defaultResources, device->GetOrCreateResourceTableDefaultsResource());

    uint64_t* rawSlots = reinterpret_cast<uint64_t*>([*mTableBuffer contents]);
    for (auto [i, defaultResource] : Enumerate(defaultResources->GetResources())) {
        uint32_t slot = uint32_t{GetAPISize() + i};
        MatchVariant(
            defaultResource,
            [&](Ref<TextureViewBase> view) {
                TextureView* texView = ToBackend(view.Get());
                id<MTLTexture> mtlTexture =
                    texView->GetTexture()->IsDestroyed() ? nil : texView->GetMTLTexture();
                rawSlots[slot] = (mtlTexture != nil) ? [mtlTexture gpuResourceID]._impl : 0;
                if (mtlTexture != nil) {
                    mDefaultResidentResources.push_back(mtlTexture);
                }
            },
            [&](Ref<SamplerBase> sampler) {
                Sampler* samp = ToBackend(sampler.Get());
                id<MTLSamplerState> mtlSampler = samp->GetMTLSamplerState();
                rawSlots[slot] = (mtlSampler != nil) ? [mtlSampler gpuResourceID]._impl : 0;
            });
    }

    return {};
}

void ResourceTable::DestroyImpl(DestroyReason reason) {
    ResourceTableBase::DestroyImpl(reason);
    mTableBuffer = nullptr;
    mSyncEvent = nullptr;
    mDefaultResidentResources.clear();
    mReferencedTextureViews.clear();
    mActiveTextures.clear();
    mResidentMTLResources.clear();
}

void ResourceTable::SetLabelImpl() {
    SetDebugName(GetDevice(), *mTableBuffer, "Dawn_ResourceTable", GetLabel());
}

MaybeError ResourceTable::ApplyPendingUpdates(
    CommandRecordingContext* commandContext,
    const absl::flat_hash_set<TextureBase*>& writableTextures) {
    return ApplyDirtySlotUpdatesWith(
        writableTextures,
        [&](const std::vector<MetadataUpdate>& metadataUpdates,
            const std::vector<ResourceDiff>& resourceDiffs,
            const absl::flat_hash_set<raw_ptr<TextureBase>>& texturesToTransition) -> MaybeError {
            if (!texturesToTransition.empty()) {
                DAWN_TRY(TransitionResources(commandContext, texturesToTransition));
            }
            if (!metadataUpdates.empty()) {
                DAWN_TRY(UpdateMetadataBuffer(commandContext, metadataUpdates));
            }
            if (!resourceDiffs.empty()) {
                DAWN_TRY(UpdateResourceBindings(resourceDiffs));
            }
            return {};
        });
}

MaybeError ResourceTable::ApplyPendingUpdates(Device* device) {
    CommandRecordingContext* commandContext =
        ToBackend(device->GetQueue())->GetPendingCommandContext();
    return ApplyPendingUpdates(commandContext, {});
}

MaybeError ResourceTable::TransitionResources(
    CommandRecordingContext* commandContext,
    const absl::flat_hash_set<raw_ptr<TextureBase>>& textures) {
    for (const auto& texture : textures) {
        Texture* textureBackend = ToBackend(texture);
        DAWN_TRY(textureBackend->EnsureSubresourceContentInitialized(
            commandContext, textureBackend->GetAllSubresources()));
        textureBackend->SynchronizeTextureBeforeUse(commandContext);
    }
    return {};
}

MaybeError ResourceTable::UpdateMetadataBuffer(CommandRecordingContext* commandContext,
                                               const std::vector<MetadataUpdate>& updates) {
    Device* device = ToBackend(GetDevice());
    Buffer* metadataBuffer = ToBackend(ResourceTableBase::GetMetadataBuffer());
    metadataBuffer->TrackUsage();
    id<MTLBuffer> dstMtlBuffer = metadataBuffer->GetMTLBuffer();

    return device->GetDynamicUploader()->WithUploadReservation(
        sizeof(uint32_t) * updates.size(), kCopyBufferToBufferOffsetAlignment,
        [&](UploadReservation reservation) -> MaybeError {
            Span<uint32_t> stagedData = ReinterpretSpan<uint32_t>(reservation.mappedData);
            id<MTLBuffer> srcMtlBuffer = ToBackend(reservation.buffer.Get())->GetMTLBuffer();

            id<MTLBlitCommandEncoder> blit = commandContext->EnsureBlit();
            for (auto [i, update] : Enumerate(updates)) {
                stagedData[i] = update.data;
                [blit copyFromBuffer:srcMtlBuffer
                         sourceOffset:reservation.offsetInBuffer + i * sizeof(uint32_t)
                             toBuffer:dstMtlBuffer
                    destinationOffset:update.offset
                                 size:sizeof(uint32_t)];
            }
            return {};
        });
}

MaybeError ResourceTable::UpdateResourceBindings(const std::vector<ResourceDiff>& diffs) {
    Device* device = ToBackend(GetDevice());
    if (mLastUsageSerial > device->GetQueue()->GetCompletedCommandSerial()) {
        NSUInteger bufferSize = [*mTableBuffer length];
        id<MTLBuffer> newBuffer =
            [device->GetMTLDevice() newBufferWithLength:bufferSize
                                                options:MTLResourceStorageModeShared];
        memcpy([newBuffer contents], [*mTableBuffer contents], bufferSize);
        mTableBuffer = AcquireNSPRef(newBuffer);
        mLastUsageSerial = kBeginningOfGPUTime;
    }

    uint64_t* rawSlots = reinterpret_cast<uint64_t*>([*mTableBuffer contents]);

    for (const ResourceDiff& diff : diffs) {
        uint32_t slot = uint32_t{diff.slot};

        // Remove old texture reference if present
        MatchVariant(
            diff.removed, [](std::monostate) {},
            [&](Ref<TextureViewBase> view) {
                auto it = mReferencedTextureViews.find(view);
                if (it != mReferencedTextureViews.end()) {
                    if (--it->second == 0) {
                        mReferencedTextureViews.erase(it);
                    }
                    mResidentResourcesDirty = true;
                }
            },
            [](Ref<SamplerBase>) {});

        // Add new resource
        MatchVariant(
            diff.added, [&](std::monostate) { rawSlots[slot] = 0; },
            [&](Ref<TextureViewBase> view) {
                TextureView* texView = ToBackend(view.Get());
                id<MTLTexture> mtlTexture =
                    texView->GetTexture()->IsDestroyed() ? nil : texView->GetMTLTexture();
                rawSlots[slot] = (mtlTexture != nil) ? [mtlTexture gpuResourceID]._impl : 0;

                if (!texView->GetTexture()->IsDestroyed()) {
                    mReferencedTextureViews[view]++;
                    mResidentResourcesDirty = true;
                }
            },
            [&](Ref<SamplerBase> sampler) {
                Sampler* samp = ToBackend(sampler.Get());
                id<MTLSamplerState> mtlSampler = samp->GetMTLSamplerState();
                rawSlots[slot] = (mtlSampler != nil) ? [mtlSampler gpuResourceID]._impl : 0;
            });
    }
    return {};
}

void ResourceTable::EncodeSync(CommandRecordingContext* commandContext) {
    Device* device = ToBackend(GetDevice());
    if (mSyncEvent == nil) {
        mSyncEvent = AcquireNSPRef([device->GetMTLDevice() newSharedEvent]);
    }
    if (mSyncEvent != nil) {
        uint64_t val = ++mSyncVal;
        id<MTLCommandBuffer> cmds = commandContext->GetCommands();
        if (cmds != nil) {
            [cmds encodeSignalEvent:*mSyncEvent value:val];
            [cmds encodeWaitForEvent:*mSyncEvent value:val];
        }
    }
}

id<MTLBuffer> ResourceTable::GetTableBuffer() const {
    return *mTableBuffer;
}

id<MTLBuffer> ResourceTable::GetMetadataBuffer() const {
    return ToBackend(ResourceTableBase::GetMetadataBuffer())->GetMTLBuffer();
}

id<MTLBuffer> ResourceTable::GetMetadataMTLBuffer() const {
    return GetMetadataBuffer();
}

void ResourceTable::RebuildResidentResources() const {
    mActiveTextures.clear();
    mResidentMTLResources.clear();

    mResidentMTLResources.insert(mResidentMTLResources.end(), mDefaultResidentResources.begin(),
                                 mDefaultResidentResources.end());

    absl::flat_hash_set<TextureBase*> seenTextures;
    for (const auto& [view, count] : mReferencedTextureViews) {
        if (count > 0 && !view->GetTexture()->IsDestroyed()) {
            TextureView* texView = ToBackend(view.Get());
            id<MTLTexture> viewMtl = texView->GetMTLTexture();
            if (viewMtl != nil) {
                mResidentMTLResources.push_back(viewMtl);
            }

            Texture* texture = ToBackend(view->GetTexture());
            if (seenTextures.insert(texture).second) {
                mActiveTextures.push_back(texture);
                Aspect aspects = texture->GetFormat().aspects;
                const Aspect kNonPlanarAspects{~(Aspect::Plane0 | Aspect::Plane1 | Aspect::Plane2)};
                for (Aspect plane :
                     {kNonPlanarAspects, Aspect::Plane0, Aspect::Plane1, Aspect::Plane2}) {
                    Aspect aspect = aspects & plane;
                    if (aspect != Aspect::None) {
                        id<MTLTexture> planeTex = texture->GetMTLTexture(aspect);
                        if (planeTex != nil && planeTex != viewMtl) {
                            mResidentMTLResources.push_back(planeTex);
                        }
                    }
                }
            }
        }
    }
    mResidentResourcesDirty = false;
}

const std::vector<TextureBase*>& ResourceTable::GetActiveTextures() const {
    if (mResidentResourcesDirty) {
        RebuildResidentResources();
    }
    return mActiveTextures;
}

void ResourceTable::MakeResourcesResident(id<MTLRenderCommandEncoder> encoder) const {
    if (mResidentResourcesDirty) {
        RebuildResidentResources();
    }
    if (!mResidentMTLResources.empty()) {
        [encoder useResources:mResidentMTLResources.data()
                        count:mResidentMTLResources.size()
                        usage:MTLResourceUsageRead
                       stages:MTLRenderStageVertex | MTLRenderStageFragment];
    }
}

void ResourceTable::MakeResourcesResident(id<MTLComputeCommandEncoder> encoder) const {
    if (mResidentResourcesDirty) {
        RebuildResidentResources();
    }
    if (!mResidentMTLResources.empty()) {
        [encoder useResources:mResidentMTLResources.data()
                        count:mResidentMTLResources.size()
                        usage:MTLResourceUsageRead];
    }
}

}  // namespace dawn::native::metal
