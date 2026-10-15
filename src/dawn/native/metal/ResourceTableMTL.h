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

#ifndef SRC_DAWN_NATIVE_METAL_RESOURCETABLEMTL_H_
#define SRC_DAWN_NATIVE_METAL_RESOURCETABLEMTL_H_

#import <Metal/Metal.h>

#include <atomic>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "src/dawn/common/NSRef.h"
#include "src/dawn/native/Error.h"
#include "src/dawn/native/IntegerTypes.h"
#include "src/dawn/native/ResourceTable.h"
#include "src/dawn/native/Texture.h"

namespace dawn::native::metal {

class CommandRecordingContext;
class Device;
class Texture;

class ResourceTable final : public ResourceTableBase {
  public:
    static ResultOrError<Ref<ResourceTable>> Create(Device* device,
                                                    const ResourceTableDescriptor* descriptor);

    MaybeError ApplyPendingUpdates(CommandRecordingContext* commandContext,
                                   const absl::flat_hash_set<TextureBase*>& writableTextures = {});
    MaybeError ApplyPendingUpdates(Device* device);
    void EncodeSync(CommandRecordingContext* commandContext);

    id<MTLBuffer> GetTableBuffer() const;
    id<MTLBuffer> GetMetadataBuffer() const;
    id<MTLBuffer> GetMetadataMTLBuffer() const;

    const std::vector<TextureBase*>& GetActiveTextures() const;

    void MakeResourcesResident(id<MTLRenderCommandEncoder> encoder) const;
    void MakeResourcesResident(id<MTLComputeCommandEncoder> encoder) const;

    void SetLastUsageSerial(ExecutionSerial serial) { mLastUsageSerial = serial; }
    ExecutionSerial GetLastUsageSerial() const { return mLastUsageSerial; }

  protected:
    void DestroyImpl(DestroyReason reason) override;
    void SetLabelImpl() override;

  private:
    ~ResourceTable() override;
    using ResourceTableBase::ResourceTableBase;
    MaybeError Initialize();

    MaybeError TransitionResources(CommandRecordingContext* commandContext,
                                   const absl::flat_hash_set<raw_ptr<TextureBase>>& textures);
    MaybeError UpdateMetadataBuffer(CommandRecordingContext* commandContext,
                                    const std::vector<MetadataUpdate>& updates);
    MaybeError UpdateResourceBindings(const std::vector<ResourceDiff>& diffs);
    void RebuildResidentResources() const;

    NSPRef<id<MTLBuffer>> mTableBuffer;

    absl::flat_hash_map<Ref<TextureViewBase>, uint32_t> mReferencedTextureViews;
    std::vector<id<MTLResource>> mDefaultResidentResources;
    mutable std::vector<TextureBase*> mActiveTextures;
    mutable std::vector<id<MTLResource>> mResidentMTLResources;
    mutable bool mResidentResourcesDirty = true;

    NSPRef<id<MTLSharedEvent>> mSyncEvent;
    std::atomic<uint64_t> mSyncVal{0};
    ExecutionSerial mLastUsageSerial = kBeginningOfGPUTime;
};

}  // namespace dawn::native::metal

#endif  // SRC_DAWN_NATIVE_METAL_RESOURCETABLEMTL_H_
