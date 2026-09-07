import numpy as np


def simulate_charged_track(
    origin,
    momentum,
    charge,
    mass=0.000511,
    b_field=2.0,
    n_steps=2000,
    t_max=8.0,
):

    origin=np.array(origin,dtype=float)
    p0=np.array(momentum,dtype=float)

    c_mm_ns=299.792458

    b_vec=np.array([0,0,b_field],dtype=float)

    def velocity(p):
        energy=np.sqrt(mass**2 + np.dot(p,p))
        return c_mm_ns*p/(energy+1e-30)


    def rhs(state):

        r=state[:3]
        p=state[3:6]

        v=velocity(p)

        beta=v/c_mm_ns

        dpdt=0.3*charge*np.cross(beta,b_vec)

        drdt=v

        return np.concatenate([drdt,dpdt])


    dt=t_max/(n_steps-1)

    state=np.concatenate([origin,p0])

    xyz=np.zeros((n_steps,3))


    for i in range(n_steps):

        xyz[i]=state[:3]


        k1=rhs(state)
        k2=rhs(state+0.5*dt*k1)
        k3=rhs(state+0.5*dt*k2)
        k4=rhs(state+dt*k3)


        state += (dt/6)*(k1+2*k2+2*k3+k4)


    return xyz

def keep_outward_branch(xyz):

    import numpy as np

    xyz=np.asarray(xyz)

    r=np.sqrt(
        xyz[:,0]**2 +
        xyz[:,1]**2
    )

    dr=np.diff(r)

    drops=0

    for i,d in enumerate(dr):

        if d < 0:
            drops += 1
        else:
            drops = 0

        if drops > 20:
            return xyz[:i+1]

    return xyz

