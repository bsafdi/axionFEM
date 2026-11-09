"""Independent axisymmetric TM0 (m=0) eigen-solver: H_phi formulation, bilinear Q1 on a graded tensor grid.
curl (1/eps curl H) = k^2 H; PEC walls = natural BC; axis H_phi = 0; optional z-symmetry plane at z=0 (even modes).
Geometry: list of axis-aligned rectangles (r0,r1,z0,z1,kind) painted in order; kind = 'metal' or eps (float).
Domain = [0,R] x [zlo,zhi]; everything outside painted rectangles is vacuum. """
import numpy as np, scipy.sparse as sp, scipy.sparse.linalg as sla
C0 = 299792458.0

def graded(breaks, hmin, hmax, growth=0.12, fine=None):
    """1D nodes: include all breaks; spacing hmin at 'fine' breakpoints (default all), growing linearly."""
    breaks = sorted(set(breaks)); fine = breaks if fine is None else fine
    fine = np.array(sorted(fine))
    xs = [breaks[0]]
    for a, b in zip(breaks[:-1], breaks[1:]):
        x = a; pts = []
        while True:
            d = np.min(np.abs(fine - x)); dn = np.min(np.abs(fine - min(b, x + hmax)))
            h = min(hmax, hmin + growth * min(d, dn))
            if x + h >= b - 0.3 * h: break
            x += h; pts.append(x)
        # rescale so the last cell is not tiny
        pts = np.array(pts)
        if len(pts):
            pts = a + (pts - a) * (b - a) / (pts[-1] + (b - pts[-1]) - a) if False else pts
        xs += list(pts) + [b]
    return np.array(xs)

def solve(R, zlo, zhi, rects, rgrid, zgrid, k_target, nev=6, sym=True, sigma=None):
    r, z = rgrid, zgrid; nr, nz = len(r), len(z)
    rc = 0.5 * (r[1:] + r[:-1]); zc = 0.5 * (z[1:] + z[:-1])
    inv_eps = np.ones((nr - 1, nz - 1)); metal = np.zeros((nr - 1, nz - 1), bool)
    RC, ZC = np.meshgrid(rc, zc, indexing='ij')
    for (r0, r1, z0, z1, kind) in rects:
        m = (RC > r0) & (RC < r1) & (ZC > z0) & (ZC < z1)
        if kind == 'metal': metal |= m
        else: inv_eps[m] = 1.0 / kind; metal[m] = False
    gp = np.array([-1, 1]) / np.sqrt(3); gw = np.array([1.0, 1.0])
    rows, cols, kv, mv = [], [], [], []
    act = np.argwhere(~metal)
    nid = lambda i, j: i * nz + j
    loc = [(0, 0), (1, 0), (0, 1), (1, 1)]
    # precompute per-cell matrices vectorised
    I, J = act[:, 0], act[:, 1]
    hr = r[I + 1] - r[I]; hz = z[J + 1] - z[J]
    Ke = np.zeros((len(I), 4, 4)); Me = np.zeros((len(I), 4, 4)); Ce = np.zeros((len(I), 4))
    for a, wa in zip(gp, gw):
        for b, wb in zip(gp, gw):
            xi, et = (a + 1) / 2, (b + 1) / 2
            rr = r[I] + xi * hr; w = wa * wb / 4 * hr * hz * rr
            N = np.array([(1 - xi) * (1 - et), xi * (1 - et), (1 - xi) * et, xi * et]).T  # (4,) const per gp
            dNr = np.array([-(1 - et), (1 - et), -et, et])[None, :] / hr[:, None]
            dNz = np.array([-(1 - xi), -xi, (1 - xi), xi])[None, :] / hz[:, None]
            Nn = np.tile(N, (len(I), 1))
            curlz = dNr + Nn / rr[:, None]
            ie = inv_eps[I, J]
            Ke += (ie * w)[:, None, None] * (dNz[:, :, None] * dNz[:, None, :] + curlz[:, :, None] * curlz[:, None, :])
            Me += w[:, None, None] * (Nn[:, :, None] * Nn[:, None, :])
            Ce += (ie * w)[:, None] * curlz
    gid = np.stack([nid(I + di, J + dj) for di, dj in loc], 1)
    K = sp.coo_matrix((Ke.ravel(), (np.repeat(gid, 4, 1).ravel(), np.tile(gid, (1, 4)).ravel())), shape=(nr * nz,) * 2).tocsr()
    M = sp.coo_matrix((Me.ravel(), (np.repeat(gid, 4, 1).ravel(), np.tile(gid, (1, 4)).ravel())), shape=(nr * nz,) * 2).tocsr()
    used = np.zeros(nr * nz, bool); used[gid.ravel()] = True
    axis = np.zeros(nr * nz, bool); axis[nid(0, np.arange(nz))] = True
    free = np.where(used & ~axis)[0]
    Kf = K[free][:, free]; Mf = M[free][:, free]
    vals, vecs = sla.eigsh(Kf, k=nev, M=Mf, sigma=k_target ** 2, which='LM')
    out = []
    # volume (cavity, metal excluded), full or half
    Vcell = np.pi * (r[I + 1] ** 2 - r[I] ** 2) * hz
    V = Vcell.sum() * (2 if sym else 1)
    for lam, v in sorted(zip(vals, vecs.T), key=lambda t: t[0]):
        u = np.zeros(nr * nz); u[free] = v
        ue = u[gid]
        Ez = np.einsum('ek,ek->e', Ce, ue)  # int (1/eps)(u_r+u/r) r dr dz per cell
        W = np.einsum('ei,eij,ej->e', ue, Ke, ue)
        f = 2
        C = (Ez.sum() * 2 * np.pi * (f if sym else 1)) ** 2 / (V * W.sum() * 2 * np.pi * (f if sym else 1))
        rec = {"f_Hz": C0 * np.sqrt(lam) / (2 * np.pi), "C": C, "u": u}
        if sigma:
            rec["Q"] = qwall(u, r, z, metal, sym, rec["f_Hz"], sigma, zhi)
        out.append(rec)
    return out, dict(nr=nr, nz=nz, ndof=len(free), V=V)

def qwall(u, r, z, metal, sym, f, sigma, zhi):
    """Q_wall = 2 int|H|^2 dV / (delta oint |H_t|^2 dA), PEC-perturbation."""
    nr, nz = len(r), len(z); U = u.reshape(nr, nz)
    mu0 = 4e-7*np.pi; delta = np.sqrt(2/(2*np.pi*f*mu0*sigma))
    gp = np.array([-1, 1])/np.sqrt(3)
    vol = 0.0; surf = 0.0
    nc_r, nc_z = metal.shape
    for i in range(nc_r):
        for j in range(nc_z):
            if metal[i, j]: continue
            hr, hz = r[i+1]-r[i], z[j+1]-z[j]
            for a in gp:
                for b in gp:
                    xi, et = (a+1)/2, (b+1)/2
                    uu = U[i,j]*(1-xi)*(1-et)+U[i+1,j]*xi*(1-et)+U[i,j+1]*(1-xi)*et+U[i+1,j+1]*xi*et
                    vol += uu**2*(r[i]+xi*hr)*hr*hz/4
            # edges: (neighbour cell index or None if outer boundary)
            def edge_r(jj, zz):  # horizontal edge at z = z[jj] spanning r[i]..r[i+1]
                return sum(((U[i,jj]*(1-(a+1)/2)+U[i+1,jj]*(a+1)/2)**2)*(r[i]+(a+1)/2*hr)*hr/2 for a in gp)
            def edge_z(ii):
                return sum(((U[ii,j]*(1-(b+1)/2)+U[ii,j+1]*(b+1)/2)**2)*r[ii]*hz/2 for b in gp)
            # bottom (z[j])
            if j == 0:
                if not sym: surf += edge_r(j, None)
            elif metal[i, j-1]: surf += edge_r(j, None)
            if j == nc_z-1 or metal[i, j+1]: surf += edge_r(j+1, None)
            if i == nc_r-1 or metal[i+1, j]: surf += edge_z(i+1)
            if i > 0 and metal[i-1, j]: surf += edge_z(i)
    return 2*vol/(delta*surf)
