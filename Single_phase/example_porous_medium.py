import time
import taichi as ti

ti.init(arch=ti.cpu)
import LBM_3D_SinglePhase_Solver as lb3dsp

time_init = time.time()
time_now = time.time()
time_pre = time.time()             


lb3d = lb3dsp.LB3D_Solver_Single_Phase(nx=103,ny=103,nz=103)

lb3d.init_geo_ply('./bone_file2.ply')
lb3d.set_bc_rho_x1(0.99)
lb3d.set_bc_rho_x0(1.0)
lb3d.init_simulation()

for iter in range(10000+1):
    lb3d.step()

    if (iter%100==0):
        
        time_pre = time_now
        time_now = time.time()
        diff_time = int(time_now-time_pre)
        elap_time = int(time_now-time_init)
        m_diff, s_diff = divmod(diff_time, 60)
        h_diff, m_diff = divmod(m_diff, 60)
        m_elap, s_elap = divmod(elap_time, 60)
        h_elap, m_elap = divmod(m_elap, 60)
        
        print('----------Time between two outputs is %dh %dm %ds; elapsed time is %dh %dm %ds----------------------' %(h_diff, m_diff, s_diff,h_elap,m_elap,s_elap))
        max_v = lb3d.get_max_v()

        print('The %dth iteration, Max Force = %f,  force_scale = %f\n\n ' %(iter, max_v,  10.0))
        
    if (iter%3==1):
        lb3d.write_ply("./out7/LB_SingelPhase_"+str(int(iter/3+1))+".ply")
            
