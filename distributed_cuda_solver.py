import os
import sys
import json
import math
import hashlib
import subprocess
import time
import numpy as np
from mpi4py import MPI
from numba import cuda, float64

TILE_DIM = 16
SHARED_DIM = 18
GLOBAL_NX = 512
GLOBAL_NY = 512
LOCAL_NX = 256
LOCAL_NY = 256
STEPS = 400
LAMBDA_DAMP = 0.15
DX = 1.0 / 513.0
DY = 1.0 / 513.0


def get_gpu_uuid():
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"],
        encoding="utf-8"
    )
    return out.strip().splitlines()[0].strip()


def derive_student_seed(github_user, gpu_uuid):
    digest = hashlib.sha256(
        f"{github_user.strip()}::{gpu_uuid}".encode("utf-8")
    ).hexdigest()
    return int(digest[:16], 16)


def init_distributed_context(github_user):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    if size != 4:
        raise RuntimeError(f"Expected exactly 4 MPI ranks (2x2 grid), got {size}")

    cuda.select_device(0)
    gpu_uuid = get_gpu_uuid() if rank == 0 else None
    gpu_uuid = comm.bcast(gpu_uuid, root=0)
    instance_seed = derive_student_seed(github_user, gpu_uuid)

    cart_comm = comm.Create_cart(
        dims=[2, 2], periods=[False, False], reorder=False
    )
    coords = cart_comm.Get_coords(rank)
    nbr_up, nbr_down = cart_comm.Shift(0, 1)
    nbr_left, nbr_right = cart_comm.Shift(1, 1)

    if rank == 0:
        print(f"MPI ranks: {size}; Cartesian grid: 2x2")
        print(f"GPU UUID: {gpu_uuid}")
        print(f"INSTANCE_SEED: {instance_seed}")
    return comm, cart_comm, rank, coords, (nbr_up, nbr_down, nbr_left, nbr_right), instance_seed


def initialize_subdomains(coords, instance_seed, neighbors):
    row_rank, col_rank = coords
    # Each rank creates only its own padded 258x258 arrays.
    gi = row_rank * LOCAL_NY + (np.arange(258) - 1)
    gj = col_rank * LOCAL_NX + (np.arange(258) - 1)
    y = (gi + 1) * DY
    x = (gj + 1) * DX
    xx, yy = np.meshgrid(x, y)

    kappa = 0.16 + 0.08 * np.sin(2.0 * np.pi * xx) * np.cos(2.0 * np.pi * yy)
    kappa = np.clip(kappa, 0.08, 0.24)

    rng = np.random.default_rng(instance_seed)
    pulse_specs = []
    for _ in range(3):
        cx = float(rng.uniform(0.2, 0.8))
        cy = float(rng.uniform(0.2, 0.8))
        amp = float(rng.uniform(0.4, 0.9))
        width = float(rng.uniform(0.035, 0.075))
        pulse_specs.append((cx, cy, amp, width))

    u = np.zeros((258, 258), dtype=np.float64)
    for cx, cy, amp, width in pulse_specs:
        u += amp * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2.0 * width ** 2))

    # A localized negative source acts as a weak sink, supporting the
    # dissipative energy trace requested in the specification.
    sx, sy = 0.52, 0.48
    source = -0.002 * np.exp(-((xx - sx) ** 2 + (yy - sy) ** 2) / (2.0 * 0.08 ** 2))

    nbr_up, nbr_down, nbr_left, nbr_right = neighbors
    if nbr_up == MPI.PROC_NULL:
        u[0, :] = 0.0
    if nbr_down == MPI.PROC_NULL:
        u[257, :] = 0.0
    if nbr_left == MPI.PROC_NULL:
        u[:, 0] = 0.0
    if nbr_right == MPI.PROC_NULL:
        u[:, 257] = 0.0

    # Outer physical boundary values are zero; owned cells remain initialized.
    return u, kappa.astype(np.float64), source.astype(np.float64)


@cuda.jit
def pack_halos_kernel(u, sendbuf):
    tid = cuda.grid(1)
    if tid < 256:
        sendbuf[tid] = u[1, tid + 1]             # top
        sendbuf[256 + tid] = u[256, tid + 1]     # bottom
        sendbuf[512 + tid] = u[tid + 1, 1]      # left
        sendbuf[768 + tid] = u[tid + 1, 256]    # right


@cuda.jit
def unpack_halos_kernel(u, recvbuf):
    tid = cuda.grid(1)
    if tid < 256:
        u[0, tid + 1] = recvbuf[tid]
        u[257, tid + 1] = recvbuf[256 + tid]
        u[tid + 1, 0] = recvbuf[512 + tid]
        u[tid + 1, 257] = recvbuf[768 + tid]


@cuda.jit
def tiled_stencil_kernel(u, kappa, source, out,
                         dx, dy, dt, lam,
                         r_start, r_end, c_start, c_end):
    smem_u = cuda.shared.array(shape=(18, 18), dtype=float64)
    smem_k = cuda.shared.array(shape=(18, 18), dtype=float64)

    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    block_i = cuda.blockIdx.y * 16 + 1
    block_j = cuda.blockIdx.x * 16 + 1
    tid = ty * 16 + tx

    # Cooperative loading of the 18x18 tile, including its one-cell apron.
    for linear in range(tid, 18 * 18, 16 * 16):
        sy = linear // 18
        sx = linear - sy * 18
        gi = block_i + sy - 1
        gj = block_j + sx - 1
        if gi < 0:
            gi = 0
        if gi > 257:
            gi = 257
        if gj < 0:
            gj = 0
        if gj > 257:
            gj = 257
        smem_u[sy, sx] = u[gi, gj]
        smem_k[sy, sx] = kappa[gi, gj]

    cuda.syncthreads()

    i = block_i + ty
    j = block_j + tx
    if (i >= r_start and i <= r_end and
            j >= c_start and j <= c_end and
            i >= 1 and i <= 256 and j >= 1 and j <= 256):
        uc = smem_u[ty + 1, tx + 1]
        uw = smem_u[ty + 1, tx]
        ue = smem_u[ty + 1, tx + 2]
        un = smem_u[ty, tx + 1]
        us = smem_u[ty + 2, tx + 1]

        kc = smem_k[ty + 1, tx + 1]
        kw = smem_k[ty + 1, tx]
        ke = smem_k[ty + 1, tx + 2]
        kn = smem_k[ty, tx + 1]
        ks = smem_k[ty + 2, tx + 1]

        flux_x = (0.5 * (ke + kc) * (ue - uc) -
                  0.5 * (kc + kw) * (uc - uw)) / (dx * dx)
        flux_y = (0.5 * (ks + kc) * (us - uc) -
                  0.5 * (kc + kn) * (uc - un)) / (dy * dy)
        reaction = -lam * uc * uc * uc
        out[i, j] = uc + dt * (flux_x + flux_y + reaction + source[i, j])


def exchange_halos_async(cart_comm, neighbors, d_u, stream_boundary):
    nbr_up, nbr_down, nbr_left, nbr_right = neighbors
    d_send = cuda.device_array(1024, dtype=np.float64)
    d_recv = cuda.device_array(1024, dtype=np.float64)
    h_send = cuda.pinned_array(1024, dtype=np.float64)
    h_recv = cuda.pinned_array(1024, dtype=np.float64)
    h_recv[:] = 0.0

    pack_halos_kernel[1, 256, stream_boundary](d_u, d_send)
    d_send.copy_to_host(h_send, stream=stream_boundary)
    stream_boundary.synchronize()

    requests = [
        cart_comm.Irecv(h_recv[0:256], source=nbr_up, tag=11),
        cart_comm.Irecv(h_recv[256:512], source=nbr_down, tag=10),
        cart_comm.Irecv(h_recv[512:768], source=nbr_left, tag=13),
        cart_comm.Irecv(h_recv[768:1024], source=nbr_right, tag=12),
        cart_comm.Isend(h_send[0:256], dest=nbr_up, tag=10),
        cart_comm.Isend(h_send[256:512], dest=nbr_down, tag=11),
        cart_comm.Isend(h_send[512:768], dest=nbr_left, tag=12),
        cart_comm.Isend(h_send[768:1024], dest=nbr_right, tag=13),
    ]
    return requests, d_recv, h_recv


def finish_halo_exchange(cart_comm, requests, d_u, d_recv, h_recv, stream_boundary):
    MPI.Request.Waitall(requests)
    d_recv.copy_to_device(h_recv, stream=stream_boundary)
    unpack_halos_kernel[1, 256, stream_boundary](d_u, d_recv)


def run_solver(cart_comm, rank, coords, neighbors, instance_seed, mode):
    u_host, k_host, s_host = initialize_subdomains(coords, instance_seed, neighbors)
    d_u = cuda.to_device(u_host)
    d_k = cuda.to_device(k_host)
    d_s = cuda.to_device(s_host)
    d_out = cuda.device_array((258, 258), dtype=np.float64)

    stream_boundary = cuda.stream()
    stream_interior = cuda.stream()
    kmax_local = float(np.max(k_host[1:257, 1:257]))
    kmax = cart_comm.allreduce(kmax_local, op=MPI.MAX)
    dt_limit = DX * DX / (4.0 * kmax)
    dt = 0.20 * dt_limit

    energy_trace = []
    residual_trace = []
    initial_host = d_u.copy_to_host()
    initial_energy_local = float(np.sum(initial_host[1:257, 1:257]) * DX * DY)
    initial_energy = cart_comm.allreduce(initial_energy_local, op=MPI.SUM)
    energy_trace.append((0, initial_energy))
    if rank == 0:
        print(f"[{mode}] step=  0 energy={initial_energy:.16e}")
    start = MPI.Wtime()

    for step in range(1, STEPS + 1):
        requests, d_recv, h_recv = exchange_halos_async(
            cart_comm, neighbors, d_u, stream_boundary
        )

        if mode == "overlapped":
            for req in requests[4:]:
                pass
            tiled_stencil_kernel[(16, 16), (16, 16), stream_interior](
                d_u, d_k, d_s, d_out, DX, DY, dt, LAMBDA_DAMP,
                2, 255, 2, 255
            )

        finish_halo_exchange(
            cart_comm, requests, d_u, d_recv, h_recv, stream_boundary
        )

        if mode == "serialized":
            tiled_stencil_kernel[(16, 16), (16, 16), stream_boundary](
                d_u, d_k, d_s, d_out, DX, DY, dt, LAMBDA_DAMP,
                1, 256, 1, 256
            )
        else:
            # Boundary ring, excluding corners in the side launches to avoid
            # updating corner cells twice.
            tiled_stencil_kernel[(16, 16), (16, 16), stream_boundary](
                d_u, d_k, d_s, d_out, DX, DY, dt, LAMBDA_DAMP,
                1, 1, 1, 256
            )
            tiled_stencil_kernel[(16, 16), (16, 16), stream_boundary](
                d_u, d_k, d_s, d_out, DX, DY, dt, LAMBDA_DAMP,
                256, 256, 1, 256
            )
            tiled_stencil_kernel[(16, 16), (16, 16), stream_boundary](
                d_u, d_k, d_s, d_out, DX, DY, dt, LAMBDA_DAMP,
                2, 255, 1, 1
            )
            tiled_stencil_kernel[(16, 16), (16, 16), stream_boundary](
                d_u, d_k, d_s, d_out, DX, DY, dt, LAMBDA_DAMP,
                2, 255, 256, 256
            )

        stream_boundary.synchronize()
        if mode == "overlapped":
            stream_interior.synchronize()

        d_u, d_out = d_out, d_u

        if step % 100 == 0:
            current = d_u.copy_to_host()
            previous = d_out.copy_to_host()
            local_energy = float(np.sum(current[1:257, 1:257]) * DX * DY)
            local_residual_sq = float(np.sum(
                (current[1:257, 1:257] - previous[1:257, 1:257]) ** 2
            ))
            global_energy = cart_comm.allreduce(local_energy, op=MPI.SUM)
            global_residual = math.sqrt(
                cart_comm.allreduce(local_residual_sq, op=MPI.SUM)
            )
            energy_trace.append((step, global_energy))
            residual_trace.append((step, global_residual))
            if rank == 0:
                print(f"[{mode}] step={step:3d} energy={global_energy:.16e} "
                      f"L2_residual={global_residual:.16e}")

    elapsed_local = MPI.Wtime() - start
    elapsed = cart_comm.allreduce(elapsed_local, op=MPI.MAX)
    final_state = d_u.copy_to_host()
    return {
        "elapsed_ms": elapsed * 1000.0,
        "energy_trace": energy_trace,
        "residual_trace": residual_trace,
        "final_state": final_state,
        "dt": dt,
        "dt_limit": dt_limit,
        "kappa_max": kmax,
    }


def ptx_audit():
    # Force compilation before inspecting generated PTX.
    a = cuda.to_device(np.zeros((258, 258), dtype=np.float64))
    k = cuda.to_device(np.ones((258, 258), dtype=np.float64) * 0.16)
    s = cuda.to_device(np.zeros((258, 258), dtype=np.float64))
    out = cuda.device_array((258, 258), dtype=np.float64)
    stream = cuda.stream()
    tiled_stencil_kernel[(16, 16), (16, 16), stream](
        a, k, s, out, DX, DY, 1e-8, LAMBDA_DAMP, 1, 256, 1, 256
    )
    stream.synchronize()
    sig = tiled_stencil_kernel.signatures[0]
    asm = tiled_stencil_kernel.inspect_asm(sig)
    if isinstance(asm, dict):
        asm_text = "\n".join(asm.values())
    else:
        asm_text = str(asm)
    counts = {
        "st_shared_f64": asm_text.count("st.shared.f64"),
        "ld_shared_f64": asm_text.count("ld.shared.f64"),
        "bar_sync": asm_text.count("bar.sync"),
    }
    return counts, asm_text


def main():
    # Remove any manifest left by an earlier failed run on rank 0.
    if MPI.COMM_WORLD.Get_rank() == 0 and os.path.exists("submission_manifest.json"):
        os.remove("submission_manifest.json")
    MPI.COMM_WORLD.Barrier()

    if len(sys.argv) < 2:
        raise RuntimeError("Usage: mpirun -np 4 python distributed_cuda_solver.py GITHUB_USERNAME")
    github_user = sys.argv[1]
    comm, cart_comm, rank, coords, neighbors, instance_seed = init_distributed_context(github_user)

    if rank == 0:
        print("\nRunning serialized mode...")
    serialized = run_solver(
        cart_comm, rank, coords, neighbors, instance_seed, "serialized"
    )

    if rank == 0:
        print("\nRunning overlapped mode...")
    overlapped = run_solver(
        cart_comm, rank, coords, neighbors, instance_seed, "overlapped"
    )

    diff = float(np.max(np.abs(
        serialized["final_state"][1:257, 1:257] -
        overlapped["final_state"][1:257, 1:257]
    )))
    global_diff = cart_comm.allreduce(diff, op=MPI.MAX)

    ptx_counts, _ = ptx_audit()
    ptx_ok = all(ptx_counts[k] > 0 for k in ptx_counts)

    if rank == 0:
        print("\n=== BENCHMARK SUMMARY ===")
        print(f"serialized_ms: {serialized['elapsed_ms']:.6f}")
        print(f"overlapped_ms: {overlapped['elapsed_ms']:.6f}")
        print(f"parity_max_diff: {global_diff:.16e}")
        print(f"PTX counts: {ptx_counts}")
        print(f"PTX audit passed: {ptx_ok}")
        print(f"kappa_max: {overlapped['kappa_max']:.16e}")
        print(f"dt: {overlapped['dt']:.16e}")
        print(f"dt_max: {overlapped['dt_limit']:.16e}")

        manifest = {
            "github_username": github_user,
            "mpi_ranks": 4,
            "cartesian_grid": [2, 2],
            "gpu_uuid": get_gpu_uuid(),
            "instance_seed": int(instance_seed),
            "steps": STEPS,
            "grid": [GLOBAL_NY, GLOBAL_NX],
            "serialized_ms": serialized["elapsed_ms"],
            "overlapped_ms": overlapped["elapsed_ms"],
            "parity_max_diff": global_diff,
            "ptx_counts": ptx_counts,
            "ptx_audit_passed": bool(ptx_ok),
            "kappa_max": overlapped["kappa_max"],
            "dt": overlapped["dt"],
            "dt_max": overlapped["dt_limit"],
            "energy_trace": overlapped["energy_trace"],
            "residual_trace": overlapped["residual_trace"],
        }
        with open("submission_manifest.json", "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
        print("Created submission_manifest.json")

    cart_comm.Barrier()
    if global_diff >= 1e-12:
        raise RuntimeError(
            f"Serialized/overlapped parity failed: max diff={global_diff}"
        )
    if not ptx_ok:
        raise RuntimeError(f"PTX audit failed: {ptx_counts}")


if __name__ == "__main__":
    main()
