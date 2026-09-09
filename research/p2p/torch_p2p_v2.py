#!/usr/bin/env python3
import torch, time, os
import torch.distributed as dist
import torch.multiprocessing as mp

def human(n):
    for u in ['B','KB','MB','GB']:
        if n<1024: return f"{n:.1f}{u}"
        n/=1024
    return f"{n:.1f}TB"

def test_p2p():
    print("="*60)
    print("1) 单进程 P2P torch.Tensor.to() vs cudaMemcpyPeer")
    print("="*60)
    n=torch.cuda.device_count()
    print(f"canAccessPeer 0->1: {torch.cuda.can_device_access_peer(0,1)}")
    for sz in [4<<10, 64<<10, 1<<20, 16<<20, 64<<20, 256<<20]:
        nelem=sz//4
        a=torch.randn(nelem, device='cuda:0', dtype=torch.float32)
        torch.cuda.synchronize(0)
        reps=1000 if sz<1<<20 else (100 if sz<16<<20 else 20)
        t0=time.time()
        for _ in range(reps):
            b=a.to('cuda:1')
        torch.cuda.synchronize(1)
        dt=time.time()-t0
        bw=sz*reps/dt/1e9
        print(f"  P2P 0->1 {human(sz):>8} bw={bw:.2f} GB/s  {dt/reps*1e6:.1f}us")
        del a,b
    # H2D
    for dev in range(n):
        h=torch.randn(1<<28, dtype=torch.float32) # 1GB /4
        d=torch.empty(1<<28, device=f'cuda:{dev}', dtype=torch.float32)
        reps=10
        t0=time.time()
        for _ in range(reps): d.copy_(h)
        torch.cuda.synchronize(dev)
        print(f"  H2D GPU{dev} bw={(1<<30)*reps/(time.time()-t0)/1e9:.2f} GB/s")
        t0=time.time()
        for _ in range(reps): h.copy_(d)
        torch.cuda.synchronize(dev)
        print(f"  D2H GPU{dev} bw={(1<<30)*reps/(time.time()-t0)/1e9:.2f} GB/s")

def test_dist(rank, world_size, backend):
    os.environ['MASTER_ADDR']='127.0.0.1'
    os.environ['MASTER_PORT']='29500'
    torch.cuda.set_device(rank)
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size, device_id=torch.device(f'cuda:{rank}'))
    dev=f'cuda:{rank}'
    sizes=[32<<10, 1<<20, 16<<20, 64<<20]
    if rank==0: print(f"\n[{backend}] world_size={world_size}")
    for sz in sizes:
        nelem=sz//4
        t=torch.randn(nelem, device=dev, dtype=torch.float32)
        for _ in range(3): dist.all_reduce(t, op=dist.ReduceOp.SUM); torch.cuda.synchronize()
        reps=100 if sz<=32<<10 else 20
        t0=time.time()
        for _ in range(reps):
            tt=t.clone()
            dist.all_reduce(tt, op=dist.ReduceOp.SUM)
        torch.cuda.synchronize()
        dt=time.time()-t0
        algbw=sz*reps/dt/1e9
        busbw=algbw*2*(world_size-1)/world_size
        if rank==0: print(f"  all_reduce {human(sz):>8} algbw={algbw:.2f} busbw={busbw:.2f} GB/s")
    for sz in sizes:
        nelem=sz//4
        t=torch.randn(nelem, device=dev, dtype=torch.float32)
        out=[torch.empty(nelem, device=dev) for _ in range(world_size)]
        for _ in range(3): dist.all_gather(out, t); torch.cuda.synchronize()
        reps=100 if sz<=32<<10 else 20
        t0=time.time()
        for _ in range(reps): dist.all_gather(out, t)
        torch.cuda.synchronize()
        dt=time.time()-t0
        algbw=sz*world_size*reps/dt/1e9
        if rank==0: print(f"  all_gather {human(sz):>8} algbw={algbw:.2f} GB/s")
    for sz in [1<<20, 16<<20]:
        nelem=sz//4
        t=torch.randn(nelem, device=dev)
        for _ in range(3): dist.broadcast(t, src=0); torch.cuda.synchronize()
        reps=20
        t0=time.time()
        for _ in range(reps): dist.broadcast(t, src=0)
        torch.cuda.synchronize()
        if rank==0: print(f"  broadcast  {human(sz):>8} algbw={sz*reps/(time.time()-t0)/1e9:.2f} GB/s")
    # P2P send/recv only for nccl
    if backend=='nccl':
        for sz in [1<<20, 16<<20, 64<<20]:
            nelem=sz//4
            t=torch.randn(nelem, device=dev)
            dist.barrier()
            if rank==0:
                t0=time.time()
                for _ in range(20): dist.send(t, dst=1)
                torch.cuda.synchronize()
                print(f"  send 0->1 {human(sz):>8} bw={sz*20/(time.time()-t0)/1e9:.2f} GB/s")
            else:
                for _ in range(20): dist.recv(t, src=0)
                torch.cuda.synchronize()
            dist.barrier()
    dist.destroy_process_group()

if __name__=="__main__":
    print(f"torch {torch.__version__} cuda {torch.version.cuda}")
    test_p2p()
    for backend in ['nccl','gloo']:
        print("\n"+"="*60)
        print(f"2) 分布式 {backend}")
        print("="*60)
        try:
            mp.spawn(test_dist, args=(2, backend), nprocs=2, join=True)
            print(f"[{backend}] PASS")
        except Exception as e:
            print(f"[{backend}] FAIL {e}")
            import traceback; traceback.print_exc()
