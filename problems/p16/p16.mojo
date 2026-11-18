# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #
from max.gpu import thread_idx, block_idx, block_dim
from max.gpu.sync import barrier
from max.gpu.host import DeviceContext
from layout import TileTensor, TensorEngine
from layout.tile_layout import row_major
from layout.tile_tensor import stack_allocation
from std.sys import argv
from std.testing import assert_equal

from harness.canary import PuzzleMemory

comptime TPB = 3
comptime SIZE = 2
comptime BLOCKS_PER_GRID = (1, 1)
comptime THREADS_PER_BLOCK = (TPB, TPB)
comptime dtype = DType.float32
comptime layout = row_major[SIZE, SIZE]()
comptime LayoutType = type_of(layout)


# ANCHOR: naive_matmul
def naive_matmul[
    size: Int,
    Engine: TensorEngine,
](
    output: TileTensor[
        mut=True, dtype, LayoutType, MutAnyOrigin, Engine=Engine
    ],
    a: TileTensor[mut=False, dtype, LayoutType, ImmutAnyOrigin, Engine=Engine],
    b: TileTensor[mut=False, dtype, LayoutType, ImmutAnyOrigin, Engine=Engine],
) where (Engine.element_size == 1):
    var row = block_dim.y * block_idx.y + thread_idx.y
    var col = block_dim.x * block_idx.x + thread_idx.x
    if row >= SIZE or col >= SIZE: 
        return 
    # FILL ME IN (roughly 6 lines)
    var shared_a = stack_allocation[dtype=dtype, address_space=.SHARED](
        row_major[SIZE, SIZE]()
    ) 
    var shared_b = stack_allocation[dtype=dtype, address_space=.SHARED](
        row_major[SIZE, SIZE]()
    ) 
    
    if row < SIZE and col < SIZE: 
        print(row,col,a[row,col])
        shared_a[row,col] = rebind[Scalar[dtype]](a[row,col]) 
        shared_b[row,col] = rebind[Scalar[dtype]](b[row,col]) 
    
    barrier() 
    # output[r][c] = sigma a[r][i] b[i][c] 
    var local_sum: Scalar[dtype] = 0
    comptime for i in range(2): 
        local_sum += shared_a[row,i] * shared_b[i,col] 
    output[row,col] = local_sum
    print(row,col,local_sum)

# ANCHOR_END: naive_matmul


# ANCHOR: single_block_matmul
def single_block_matmul[
    size: Int,
    Engine: TensorEngine,
](
    output: TileTensor[
        mut=True, dtype, LayoutType, MutAnyOrigin, Engine=Engine
    ],
    a: TileTensor[mut=False, dtype, LayoutType, ImmutAnyOrigin, Engine=Engine],
    b: TileTensor[mut=False, dtype, LayoutType, ImmutAnyOrigin, Engine=Engine],
) where (Engine.element_size == 1):
    var row = block_dim.y * block_idx.y + thread_idx.y
    var col = block_dim.x * block_idx.x + thread_idx.x
    if row >= SIZE or col >= SIZE: 
        return 
    # FILL ME IN (roughly 6 lines)
    var shared_a = stack_allocation[dtype=dtype, address_space=.SHARED](
        row_major[SIZE, SIZE]()
    ) 
    var shared_b = stack_allocation[dtype=dtype, address_space=.SHARED](
        row_major[SIZE, SIZE]()
    ) 
    
    if row < SIZE and col < SIZE: 
        print(row,col,a[row,col])
        shared_a[row,col] = rebind[Scalar[dtype]](a[row,col]) 
        shared_b[row,col] = rebind[Scalar[dtype]](b[row,col]) 
    
    barrier() 
    # output[r][c] = sigma a[r][i] b[i][c] 
    var local_sum: Scalar[dtype] = 0
    comptime for i in range(2): 
        local_sum += shared_a[row,i] * shared_b[i,col] 
    output[row,col] = local_sum


# ANCHOR_END: single_block_matmul


comptime SIZE_TILED = 9
comptime BLOCKS_PER_GRID_TILED = (3, 3)  # each block covers 3x3 elements
comptime THREADS_PER_BLOCK_TILED = (TPB, TPB)
comptime layout_tiled = row_major[SIZE_TILED, SIZE_TILED]()
comptime LayoutTiledType = type_of(layout_tiled)


# 한 스테르듣 (3,3) 짜리 들고 있다는거고 

# ANCHOR: matmul_tiled
def matmul_tiled[
    size: Int,
    Engine: TensorEngine,
](
    output: TileTensor[
        mut=True, dtype, LayoutTiledType, MutAnyOrigin, Engine=Engine
    ],
    a: TileTensor[
        mut=False, dtype, LayoutTiledType, ImmutAnyOrigin, Engine=Engine
    ],
    b: TileTensor[
        mut=False, dtype, LayoutTiledType, ImmutAnyOrigin, Engine=Engine
    ],
) where (Engine.element_size == 1):
    var local_row = thread_idx.y
    var local_col = thread_idx.x
    var tiled_row = block_idx.y * TPB + local_row
    var tiled_col = block_idx.x * TPB + local_col
    if tiled_row >= SIZE_TILED or tiled_col >= SIZE_TILED: 
        return 
    # FILL ME IN (roughly 20 lines)
    # 
    var shared_a = stack_allocation[dtype=dtype, address_space=.SHARED](
        row_major[TPB, TPB]()
    ) 
    var shared_b = stack_allocation[dtype=dtype, address_space=.SHARED](
        row_major[TPB, TPB]()
    ) 
    var local_sum: Scalar[dtype] = 0 
    for k in range(3): 
        # a[row][k] * b[k][col] 
        shared_a[local_row,local_col] = rebind[Scalar[dtype]](a[tiled_row,local_col + k*3]) 
        shared_b[local_row,local_col] = rebind[Scalar[dtype]](b[local_row+k*3,tiled_col])
        barrier() 
        for t in range(3): 
            local_sum += shared_a[local_row,t] * shared_b[t,local_col] 
        barrier()
    
    output[tiled_row,tiled_col] = local_sum

# ANCHOR_END: matmul_tiled


def main() raises:
    with DeviceContext() as ctx:
        var mem = PuzzleMemory[dtype](ctx)
        var size = (
            SIZE_TILED if argv()[1] == "--idiomatic-tiled"
            or argv()[1] == "--tiled" else SIZE
        )
        var out = mem.output(size * size)
        var inp1 = ctx.enqueue_create_buffer[dtype](size * size)
        inp1.enqueue_fill(0)
        var inp2 = ctx.enqueue_create_buffer[dtype](size * size)
        inp2.enqueue_fill(0)
        var expected = ctx.enqueue_create_host_buffer[dtype](size * size)
        expected.enqueue_fill(0)

        with inp1.map_to_host() as inp1_host, inp2.map_to_host() as inp2_host:
            for row in range(size):
                for col in range(size):
                    var val = row * size + col
                    # row major: placing elements row by row
                    inp1_host[row * size + col] = Scalar[dtype](val)
                    inp2_host[row * size + col] = Scalar[dtype](
                        2.0 * Float64(val)
                    )

            # inp1 @ inp2
            for i in range(size):
                for j in range(size):
                    for k in range(size):
                        expected[i * size + j] += (
                            inp1_host[i * size + k] * inp2_host[k * size + j]
                        )

        var out_tensor = TileTensor(out, layout)
        var a_tensor = TileTensor(inp1, layout)
        var b_tensor = TileTensor(inp2, layout)

        if argv()[1] == "--naive":
            comptime kernel = naive_matmul[SIZE, out_tensor.Engine]
            ctx.enqueue_function[kernel](
                out_tensor,
                a_tensor,
                b_tensor,
                grid_dim=BLOCKS_PER_GRID,
                block_dim=THREADS_PER_BLOCK,
            )
        elif argv()[1] == "--single-block":
            comptime kernel = single_block_matmul[SIZE, out_tensor.Engine]
            ctx.enqueue_function[kernel](
                out_tensor,
                a_tensor,
                b_tensor,
                grid_dim=BLOCKS_PER_GRID,
                block_dim=THREADS_PER_BLOCK,
            )
        elif argv()[1] == "--tiled":
            # Need to update the layout of the tensors to the tiled layout
            var out_tensor_tiled = TileTensor(out, layout_tiled)
            var a_tensor_tiled = TileTensor(inp1, layout_tiled)
            var b_tensor_tiled = TileTensor(inp2, layout_tiled)

            comptime kernel = matmul_tiled[SIZE_TILED, out_tensor_tiled.Engine]
            ctx.enqueue_function[kernel](
                out_tensor_tiled,
                a_tensor_tiled,
                b_tensor_tiled,
                grid_dim=BLOCKS_PER_GRID_TILED,
                block_dim=THREADS_PER_BLOCK_TILED,
            )
        else:
            raise Error(
                "Invalid option. Choose among the available flags: --naive,"
                " --single-block, --tiled, --idiomatic-tiled"
            )

        ctx.synchronize()

        with out.map_to_host() as out_host:
            print("out:", out_host)
            print("expected:", expected)
            for col in range(size):
                for row in range(size):
                    assert_equal(
                        out_host[col * size + row], expected[col * size + row]
                    )
        mem.verify()
        print("Puzzle 16 complete ✅")
