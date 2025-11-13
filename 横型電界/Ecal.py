#!/usr/bin/env python3
"""
Generate a processed NPZ file containing:
  - node coordinates (x[], y[], z[])
  - potential V[i,j,k]
  - recomputed E-field from V (Ex, Ey, Ez) using nonuniform central differences
Works with OpenSTF log files (2dz***.log) located in the same folder.
"""
import numpy as np
import glob
import re

# ---------------------------------------------
# 1) Load OpenSTF logs and merge into 3D arrays
# ---------------------------------------------
def load_openstf_logs():
    files = sorted(glob.glob("2dz*.log"))
    if not files:
        raise FileNotFoundError("No 2dz*.log found in this folder.")

    zs = []
    data_blocks = []

    pattern = re.compile(r"2dz([0-9.]+)\.log")

    for fn in files:
        m = pattern.search(fn)
        if not m:
            continue
        zpos = float(m.group(1))
        zs.append(zpos)

        # ---- ここを修正 ----
        # ヘッダー行（No. x y ...）をスキップする
        arr = np.genfromtxt(
            fn,
            comments=None,
            skip_header=1,   # 1行だけスキップ
            invalid_raise=False
        )
        # --------------------

        # 空行対策：NaN 行を除外
        arr = arr[~np.isnan(arr).any(axis=1)]
        data_blocks.append(arr)

    Z = np.array(zs)
    Z_sorted = np.sort(Z)

    base = data_blocks[0]
    X = np.unique(base[:, 0])
    Y = np.unique(base[:, 1])
    Nx, Ny, Nz = len(X), len(Y), len(Z_sorted)

    V = np.zeros((Nz, Ny, Nx))

    for i, zval in enumerate(Z_sorted):
        idx = np.where(Z == zval)[0][0]
        block = data_blocks[idx]
        for row in block:
            x, y, v = row[0], row[1], row[2]
            ix = np.where(X == x)[0][0]
            iy = np.where(Y == y)[0][0]
            V[i, iy, ix] = v

    return X, Y, Z_sorted, V


# -------------------------------------------------
# 2) Compute gradient on nonuniform grid (central)
# -------------------------------------------------
def diff_nonuniform(f, coords, axis):
    f = np.asarray(f)
    coords = np.asarray(coords)
    df = np.zeros_like(f)

    if axis == 0:  # z
        for k in range(1, len(coords)-1):
            dz1 = coords[k]   - coords[k-1]
            dz2 = coords[k+1] - coords[k]
            w1 = dz2/(dz1*(dz1+dz2))
            w2 = - (dz1+dz2)/(dz1*dz2)
            w3 = dz1/(dz2*(dz1+dz2))
            df[k,:,:] = w1*f[k-1,:,:] + w2*f[k,:,:] + w3*f[k+1,:,:]
        df[0,:,:]  = (f[1,:,:] - f[0,:,:])/(coords[1] - coords[0])
        df[-1,:,:] = (f[-1,:,:] - f[-2,:,:])/(coords[-1] - coords[-2])
    elif axis == 1:  # y
        for j in range(1, len(coords)-1):
            dy1 = coords[j]   - coords[j-1]
            dy2 = coords[j+1] - coords[j]
            w1 = dy2/(dy1*(dy1+dy2))
            w2 = -(dy1+dy2)/(dy1*dy2)
            w3 = dy1/(dy2*(dy1+dy2))
            df[:,j,:] = w1*f[:,j-1,:] + w2*f[:,j,:] + w3*f[:,j+1,:]
        df[:,0,:]  = (f[:,1,:] - f[:,0,:])/(coords[1] - coords[0])
        df[:,-1,:] = (f[:,-1,:] - f[:,-2,:])/(coords[-1] - coords[-2])
    else:  # x
        for i in range(1, len(coords)-1):
            dx1 = coords[i]   - coords[i-1]
            dx2 = coords[i+1] - coords[i]
            w1 = dx2/(dx1*(dx1+dx2))
            w2 = -(dx1+dx2)/(dx1*dx2)
            w3 = dx1/(dx2*(dx1+dx2))
            df[:,:,i] = w1*f[:,:,i-1] + w2*f[:,:,i] + w3*f[:,:,i+1]
        df[:,:,0]  = (f[:,:,1] - f[:,:,0])/(coords[1] - coords[0])
        df[:,:,-1] = (f[:,:,-1] - f[:,:,-2])/(coords[-1] - coords[-2])

    return df

# ---------------------------------------------
# 3) Main
# ---------------------------------------------
def main():
    X, Y, Z, V = load_openstf_logs()
    dVdx = diff_nonuniform(V, X, axis=2)
    dVdy = diff_nonuniform(V, Y, axis=1)
    dVdz = diff_nonuniform(V, Z, axis=0)

    Ex = -dVdx
    Ey = -dVdy
    Ez = -dVdz

    np.savez("field_processed.npz", X=X, Y=Y, Z=Z, V=V, Ex=Ex, Ey=Ey, Ez=Ez)
    print("Saved field_processed.npz with coordinates, V and recomputed E fields.")

if __name__ == "__main__":
    main()
