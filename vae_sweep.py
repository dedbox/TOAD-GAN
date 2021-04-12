import subprocess

levels = ['lvl_1-1.txt'] #, 'lvl_1-3.txt']

N = 5

# for level in levels:
#     for L in range(0, 49, N):
#         print("=" * 80)
#         print(f'I={level}    H=24    L={list(range(L+1,L+N+1))}')
#         procs = [subprocess.Popen(f"python VAE_patches2.py --input-name {level} --hidden-dim 24 --latent-dim {L+i+1} --vae-show true --vae-save true", shell=True) for i in range(N)]
#         for p in procs:
#             p.wait()

for level in levels:
    for H in range(0, 50, N):
        print("=" * 80)
        print(f'I={level}    H={list(range(H+1,H+N+1))}    L=7')
        procs = [subprocess.Popen(f"python VAE_patches2.py --input-name {level} --latent-dim 7 --hidden-dim {H+i+1} --vae-show true --vae-save true", shell=True) for i in range(N)]
        for p in procs:
            p.wait()
