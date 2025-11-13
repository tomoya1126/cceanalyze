#!/usr/bin/env python3
"""
Full validation + visualization tool for field_processed.npz

Performs:
 - E vs -∇V (RMS abs/rel error)
 - curl(E)
 - Laplacian(V)
 - |E| statistics
 - Slice visualizations: V, |E|, curl(E), Laplacian(V)
 - Quiver plot on XY slice
 - Saves all images to ./figures/
"""

import numpy as np
import matplotlib.pyplot as plt
import os

# ----------------------------
# Utility: nonuniform derivative
# ----------------------------
def diff_nonuniform(f, x):
    df = np.zeros_like(f)
    N = len(x)
    for i in range(N):
        if i == 0:
            df[i] = (f[i+1] - f[i]) / (x[i+1] - x[i])
        elif i == N-1:
            df[i] = (f[i] - f[i-1]) / (x[i] - x[i-1])
        else:
            dxp = x[i+1] - x[i]
            dxm = x[i] - x[i-1]
            df[i] = (dxp * (f[i] - f[i-1]) + dxm * (f[i+1] - f[i])) / (dxp * dxm * (dxp + dxm))
    return df

# ----------------------------
# Visualization helper
# ----------------------------
def plot2d(field, X, Y, title, fname):
    plt.figure(figsize=(6,5))
    plt.imshow(field, origin="lower",
               extent=[X.min(), X.max(), Y.min(), Y.max()],
               aspect="auto")
    plt.colorbar()
    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.tight_layout()
    plt.savefig(fname)
    plt.close()

def quiver_plot(Ex, Ey, X, Y, title, fname, step=4):
    xx, yy = np.meshgrid(X, Y)
    plt.figure(figsize=(6,5))
    plt.quiver(xx[::step,::step], yy[::step,::step],
               Ex[::step,::step], Ey[::step,::step],
               color="blue", alpha=0.7)
    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.tight_layout()
    plt.savefig(fname)
    plt.close()

# ----------------------------
# Main analysis
# ----------------------------
def main():
    if not os.path.exists("field_processed.npz"):
        print("ERROR: field_processed.npz not found")
        return

    os.makedirs("figures", exist_ok=True)

    print("Loading field_processed.npz ...")
    data = np.load("field_processed.npz")

    X = data["X"]
    Y = data["Y"]
    Z = data["Z"]
    V = data["V"]
    Ex = data["Ex"]
    Ey = data["Ey"]
    Ez = data["Ez"]

    Nz, Ny, Nx = V.shape

    print("=== FULL VALIDATION START ===\n")

    # ----------------------------
    # 1) E vs -∇V
    # ----------------------------
    print("[Step 1] Computing -∇V ...")

    dVdx = np.zeros_like(V)
    dVdy = np.zeros_like(V)
    dVdz = np.zeros_like(V)

    for k in range(Nz):
        for j in range(Ny): dVdx[k,j,:] = diff_nonuniform(V[k,j,:], X)
    for k in range(Nz):
        for i in range(Nx): dVdy[k,:,i] = diff_nonuniform(V[k,:,i], Y)
    for j in range(Ny):
        for i in range(Nx): dVdz[:,j,i] = diff_nonuniform(V[:,j,i], Z)

    Ex_err = Ex + dVdx
    Ey_err = Ey + dVdy
    Ez_err = Ez + dVdz

    rms = lambda a: np.sqrt(np.mean(a*a))

    print("  RMS abs error:")
    print(f"    Ex = {rms(Ex_err):.3e}")
    print(f"    Ey = {rms(Ey_err):.3e}")
    print(f"    Ez = {rms(Ez_err):.3e}")

    print("\n  RMS rel error:")
    print(f"    Ex = {rms(Ex_err)/rms(Ex):.3e}")
    print(f"    Ey = {rms(Ey_err)/rms(Ey):.3e}")
    print(f"    Ez = {rms(Ez_err)/rms(Ez):.3e}")

    # ----------------------------
    # 2) curl(E)
    # ----------------------------
    print("\n[Step 2] Computing curl(E) ...")

    curlx = np.zeros_like(V)
    curly = np.zeros_like(V)
    curlz = np.zeros_like(V)

    # ∂Ez/∂y - ∂Ey/∂z
    for k in range(Nz):
        for i in range(Nx):
            dEzdy = diff_nonuniform(Ez[k,:,i], Y)
            curlx[k,:,i] = dEzdy
    for j in range(Ny):
        for i in range(Nx):
            dEydz = diff_nonuniform(Ey[:,j,i], Z)
            curlx[:,j,i] -= dEydz

    # ∂Ex/∂z - ∂Ez/∂x
    for j in range(Ny):
        for i in range(Nx):
            dExdz = diff_nonuniform(Ex[:,j,i], Z)
            curly[:,j,i] = dExdz
    for k in range(Nz):
        for j in range(Ny):
            dEzdx = diff_nonuniform(Ez[k,j,:], X)
            curly[k,j,:] -= dEzdx

    # ∂Ey/∂x - ∂Ex/∂y
    for k in range(Nz):
        for j in range(Ny):
            dEydx = diff_nonuniform(Ey[k,j,:], X)
            curlz[k,j,:] = dEydx
    for k in range(Nz):
        for i in range(Nx):
            dExdy = diff_nonuniform(Ex[k,:,i], Y)
            curlz[k,:,i] -= dExdy

    curl_mag = np.sqrt(curlx**2 + curly**2 + curlz**2)
    print(f"  RMS(|curl E|) = {rms(curl_mag):.3e}")

    # ----------------------------
    # 3) Laplacian(V)
    # ----------------------------
    print("\n[Step 3] Computing Laplacian(V) ...")
    lapV = np.zeros_like(V)

    for k in range(Nz):
        for j in range(Ny): lapV[k,j,:] += diff_nonuniform(dVdx[k,j,:], X)
    for k in range(Nz):
        for i in range(Nx): lapV[k,:,i] += diff_nonuniform(dVdy[k,:,i], Y)
    for j in range(Ny):
        for i in range(Nx): lapV[:,j,i] += diff_nonuniform(dVdz[:,j,i], Z)

    print(f"  RMS(∇²V) = {rms(lapV):.3e}")

    # ----------------------------
    # 4) Statistics
    # ----------------------------
    print("\n[Step 4] E field statistics ...")
    Emag = np.sqrt(Ex**2 + Ey**2 + Ez**2)
    print(f"  E max  = {np.max(Emag):.3e}")
    print(f"  E mean = {np.mean(Emag):.3e}")
    p50, p90, p99 = np.percentile(Emag,[50,90,99])
    print(f"  percentiles = 50%:{p50:.3e}, 90%:{p90:.3e}, 99%:{p99:.3e}")

    # ----------------------------
    # 5) Visualization
    # ----------------------------
    print("\n[Step 5] Generating slice images ...")
    midz = Nz//2

    plot2d(V[midz,:,:], X, Y, "Potential V (mid z)", "figures/V_xy.png")
    plot2d(Emag[midz,:,:], X, Y, "|E| (mid z)", "figures/E_xy.png")
    plot2d(curl_mag[midz,:,:], X, Y, "curl|E| (mid z)", "figures/curl_xy.png")
    plot2d(lapV[midz,:,:], X, Y, "Laplacian V (mid z)", "figures/lapV_xy.png")

    quiver_plot(Ex[midz,:,:], Ey[midz,:,:], X, Y,
                "Electric field vectors (XY slice)", "figures/quiver_xy.png")

    print("\nAll images saved under ./figures/")

    print("\n=== FULL VALIDATION COMPLETED ===")


if __name__ == "__main__":
    main()
