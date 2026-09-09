#!/usr/bin/env python3
"""
torch手写 P2P + 分布式测试 for dual 90HX
1) 单进程双卡: torch.cuda peer copy (tensor.to), 测 P2P 带宽
2) 多进程分布式: gloo/nccl all_reduce, all_gather, broadcast, send/recv
"""
import torch
import time
import os
import torch.distributed as dist
import torch.multiprocessing as mp

def human(n):
    for unit in ['B','KB','MB','GB']:
        if n < 1024: return f"{n:.1f}{unit}"
        n/=1024
    return f"{n:.1f}TB"

def test_p2p_single_process():
    print("="*60)
    print("1) 单进程 P2P cudaMemcpyPeer via torch.Tensor.to()")
    print("="*60)
    n = torch.cuda.device_count()
    print(f"devices={n} { [torch.cuda.get_device_name(i) for i in range(n)] }")
    for src in range(n):
        for dst in range(n):
            if src==dst: continue
            for sz in [4<<10, 64<<10, 1<<20, 16<<20, 64<<20, 256<<20]:
                # float32 element =4B
                nelem = sz // 4
                a = torch.randn(nelem, device=f'cuda:{src}', dtype=torch.float32)
                torch.cuda.synchronize(src)
                # warmup
                for _ in range(3):
                    b = a.to(f'cuda:{dst}')
                    torch.cuda.synchronize(dst)
                reps = 1000 if sz< (1<<20) else (100 if sz < (16<<20) else 20)
                t0 = time.time()
                for _ in range(reps):
                    b = a.to(f'cuda:{dst}')
                torch.cuda.synchronize(dst)
                dt = time.time()-t0
                bw = sz*reps/dt/1e9
                if sz==256<<20:
                    print(f"  P2P {src}->{dst} {human(sz)} reps={reps} bw={bw:.2f} GB/s")
                del a, b
            break
        break
    # 详细 sizes 0->1
    print("\n  P2P 0->1 detailed sizes:")
    for sz in [4<<10, 64<<10, 1<<20, 16<<20, 64<<20, 256<<20]:
        nelem = sz // 4
        a = torch.randn(nelem, device='cuda:0', dtype=torch.float32)
        torch.cuda.synchronize(0)
        reps = 1000 if sz< (1<<20) else (100 if sz < (16<<20) else 20)
        t0=time.time()
        for _ in range(reps):
            b = a.to('cuda:1')
        torch.cuda.synchronize(1)
        dt=time.time()-t0
        bw=sz*reps/dt/1e9
        print(f"    {human(sz):>8} reps={reps:4d} bw={bw:.2f} GB/s  time={dt/reps*1e6:.1f} us")

    print("\n  H2D / D2H via torch:")
    for dev in range(n):
        for sz in [1<<30]: # 1GB
            nelem=sz//4
            h=torch.randn(nelem, dtype=torch.float32) # cpu
            d=torch.empty(nelem, device=f'cuda:{dev}', dtype=torch.float32)
            # H2D
            for _ in range(3): d.copy_(h, non_blocking=False)
            torch.cuda.synchronize(dev)
            reps=10
            t0=time.time()
            for _ in range(reps): d.copy_(h)
            torch.cuda.synchronize(dev)
            dt=time.time()-t0
            print(f"    H2D GPU{dev} {human(sz)} bw={sz*reps/dt/1e9:.2f} GB/s")
            # D2H
            t0=time.time()
            for _ in range(reps): h.copy_(d)
            torch.cuda.synchronize(dev)
            dt=time.time()-t0
            print(f"    D2H GPU{dev} {human(sz)} bw={sz*reps/dt/1e9:.2f} GB/s")

def test_distributed(rank, world_size, backend, results):
    os.environ['MASTER_ADDR']='127.0.0.1'
    os.environ['MASTER_PORT']='29500'
    torch.cuda.set_device(rank)
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
    device=f'cuda:{rank}'
    # all_reduce
    sizes=[32*1024, 1<<20, 16<<20, 64<<20] # 32KB 1MB 16MB 64MB
    if rank==0: print(f"\n[{backend}] rank {rank}/{world_size} testing...")
    for sz in sizes:
        nelem=sz//4
        t=torch.randn(nelem, device=device, dtype=torch.float32)
        # warmup
        for _ in range(3): dist.all_reduce(t, op=dist.ReduceOp.SUM); torch.cuda.synchronize()
        reps=20
        if sz<=32*1024: reps=100
        t0=time.time()
        for _ in range(reps):
            # need fresh tensor each iter to avoid in-place accumulation blowup? but all_reduce sums in place, we clone
            tt=t.clone()
            dist.all_reduce(tt, op=dist.ReduceOp.SUM)
        torch.cuda.synchronize()
        dt=time.time()-t0
        # busbw = 2*(n-1)/n * size / time  for all_reduce? simplified algbw = size/time
        algbw=sz*reps/dt/1e9
        busbw = algbw * 2*(world_size-1)/world_size if world_size>1 else algbw
        if rank==0:
            print(f"  [{backend}] all_reduce {human(sz):>8} algbw={algbw:.2f} GB/s busbw={busbw:.2f} GB/s time={dt/reps*1e6:.1f} us")
            results.append((backend, "all_reduce", sz, algbw, busbw))
    # all_gather
    for sz in sizes:
        nelem=sz//4
        t=torch.randn(nelem, device=device, dtype=torch.float32)
        out=[torch.empty(nelem, device=device, dtype=torch.float32) for _ in range(world_size)]
        for _ in range(3): dist.all_gather(out, t); torch.cuda.synchronize()
        reps=20
        if sz<=32*1024: reps=100
        t0=time.time()
        for _ in range(reps): dist.all_gather(out, t)
        torch.cuda.synchronize()
        dt=time.time()-t0
        algbw=sz*world_size*reps/dt/1e9 # total gathered
        if rank==0:
            print(f"  [{backend}] all_gather {human(sz):>8} algbw={algbw:.2f} GB/s time={dt/reps*1e6:.1f} us")
    # broadcast
    for sz in [1<<20, 16<<20]:
        nelem=sz//4
        t=torch.randn(nelem, device=device, dtype=torch.float32)
        for _ in range(3): dist.broadcast(t, src=0); torch.cuda.synchronize()
        reps=20
        t0=time.time()
        for _ in range(reps): dist.broadcast(t, src=0)
        torch.cuda.synchronize()
        dt=time.time()-t0
        algbw=sz*reps/dt/1e9
        if rank==0:
            print(f"  [{backend}] broadcast  {human(sz):>8} algbw={algbw:.2f} GB/s")
    # send/recv P2P
    if world_size==2:
        for sz in [1<<20, 16<<20, 64<<20]:
            nelem=sz//4
            t=torch.randn(nelem, device=device, dtype=torch.float32)
            if rank==0:
                for _ in range(3): dist.send(t, dst=1); torch.cuda.synchronize()
                reps=20
                t0=time.time()
                for _ in range(reps): dist.send(t, dst=1)
                torch.cuda.synchronize()
                dt=time.time()-t0
                print(f"  [{backend}] send 0->1 {human(sz):>8} bw={sz*reps/dt/1e9:.2f} GB/s")
            else:
                for _ in range(3): dist.recv(t, src=0); torch.cuda.synchronize()
                reps=20
                for _ in range(reps): dist.recv(t, src=0)
                torch.cuda.synchronize()
            dist.barrier()
    dist.destroy_process_group()

if __name__=="__main__":
    print(f"torch {torch.__version__} cuda {torch.version.cuda} devices {torch.cuda.device_count()}")
    print(f"canAccessPeer 0->1: {torch.cuda.can_device_access_peer(0,1) if hasattr(torch.cuda,'can_device_access_peer') else 'unknown'}")
    test_p2p_single_process()
    for backend in ['gloo','nccl']:
        print("\n"+"="*60)
        print(f"2) 分布式 {backend} world_size=2")
        print("="*60)
        try:
            mp.spawn(test_distributed, args=(2, backend, []), nprocs=2, join=True)
            print(f"[{backend}] PASS")
        except Exception as e:
            print(f"[{backend}] FAIL: {e}")
            import traceback; traceback.print_exc()
