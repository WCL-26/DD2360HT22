#include "Particles.h"
#include "Alloc.h"
#include <cuda.h>
#include <cuda_runtime.h>


/** allocate particle arrays */
void particle_allocate(struct parameters* param, struct particles* part, int is)
{
    
    // set species ID
    part->species_ID = is;
    // number of particles
    part->nop = param->np[is];
    // maximum number of particles
    part->npmax = param->npMax[is];
    
    // choose a different number of mover iterations for ions and electrons
    if (param->qom[is] < 0){  //electrons
        part->NiterMover = param->NiterMover;
        part->n_sub_cycles = param->n_sub_cycles;
    } else {                  // ions: only one iteration
        part->NiterMover = 1;
        part->n_sub_cycles = 1;
    }
    
    // particles per cell
    part->npcelx = param->npcelx[is];
    part->npcely = param->npcely[is];
    part->npcelz = param->npcelz[is];
    part->npcel = part->npcelx*part->npcely*part->npcelz;
    
    // cast it to required precision
    part->qom = (FPpart) param->qom[is];
    
    long npmax = part->npmax;
    
    // initialize drift and thermal velocities
    // drift
    part->u0 = (FPpart) param->u0[is];
    part->v0 = (FPpart) param->v0[is];
    part->w0 = (FPpart) param->w0[is];
    // thermal
    part->uth = (FPpart) param->uth[is];
    part->vth = (FPpart) param->vth[is];
    part->wth = (FPpart) param->wth[is];
    
    
    //////////////////////////////
    /// ALLOCATION PARTICLE ARRAYS
    //////////////////////////////
    part->x = new FPpart[npmax];
    part->y = new FPpart[npmax];
    part->z = new FPpart[npmax];
    // allocate velocity
    part->u = new FPpart[npmax];
    part->v = new FPpart[npmax];
    part->w = new FPpart[npmax];
    // allocate charge = q * statistical weight
    part->q = new FPinterp[npmax];
    
}
/** deallocate */
void particle_deallocate(struct particles* part)
{
    // deallocate particle variables
    delete[] part->x;
    delete[] part->y;
    delete[] part->z;
    delete[] part->u;
    delete[] part->v;
    delete[] part->w;
    delete[] part->q;
}

/** particle mover */
int mover_PC(struct particles* part, struct EMfield* field, struct grid* grd, struct parameters* param)
{
    // print species and subcycling
    std::cout << "***  MOVER with SUBCYCLYING "<< param->n_sub_cycles << " - species " << part->species_ID << " ***" << std::endl;
 
    // auxiliary variables
    FPpart dt_sub_cycling = (FPpart) param->dt/((double) part->n_sub_cycles);
    FPpart dto2 = .5*dt_sub_cycling, qomdt2 = part->qom*dto2/param->c;
    FPpart omdtsq, denom, ut, vt, wt, udotb;
    
    // local (to the particle) electric and magnetic field
    FPfield Exl=0.0, Eyl=0.0, Ezl=0.0, Bxl=0.0, Byl=0.0, Bzl=0.0;
    
    // interpolation densities
    int ix,iy,iz;
    FPfield weight[2][2][2];
    FPfield xi[2], eta[2], zeta[2];
    
    // intermediate particle position and velocity
    FPpart xptilde, yptilde, zptilde, uptilde, vptilde, wptilde;
    
    // start subcycling
    for (int i_sub=0; i_sub <  part->n_sub_cycles; i_sub++){
        // move each particle with new fields
        for (int i=0; i <  part->nop; i++){
            xptilde = part->x[i];
            yptilde = part->y[i];
            zptilde = part->z[i];
            // calculate the average velocity iteratively
            for(int innter=0; innter < part->NiterMover; innter++){
                // interpolation G-->P
                ix = 2 +  int((part->x[i] - grd->xStart)*grd->invdx);
                iy = 2 +  int((part->y[i] - grd->yStart)*grd->invdy);
                iz = 2 +  int((part->z[i] - grd->zStart)*grd->invdz);
                
                // calculate weights
                xi[0]   = part->x[i] - grd->XN[ix - 1][iy][iz];
                eta[0]  = part->y[i] - grd->YN[ix][iy - 1][iz];
                zeta[0] = part->z[i] - grd->ZN[ix][iy][iz - 1];
                xi[1]   = grd->XN[ix][iy][iz] - part->x[i];
                eta[1]  = grd->YN[ix][iy][iz] - part->y[i];
                zeta[1] = grd->ZN[ix][iy][iz] - part->z[i];
                for (int ii = 0; ii < 2; ii++)
                    for (int jj = 0; jj < 2; jj++)
                        for (int kk = 0; kk < 2; kk++)
                            weight[ii][jj][kk] = xi[ii] * eta[jj] * zeta[kk] * grd->invVOL;
                
                // set to zero local electric and magnetic field
                Exl=0.0, Eyl = 0.0, Ezl = 0.0, Bxl = 0.0, Byl = 0.0, Bzl = 0.0;
                
                for (int ii=0; ii < 2; ii++)
                    for (int jj=0; jj < 2; jj++)
                        for(int kk=0; kk < 2; kk++){
                            Exl += weight[ii][jj][kk]*field->Ex[ix- ii][iy -jj][iz- kk ];
                            Eyl += weight[ii][jj][kk]*field->Ey[ix- ii][iy -jj][iz- kk ];
                            Ezl += weight[ii][jj][kk]*field->Ez[ix- ii][iy -jj][iz -kk ];
                            Bxl += weight[ii][jj][kk]*field->Bxn[ix- ii][iy -jj][iz -kk ];
                            Byl += weight[ii][jj][kk]*field->Byn[ix- ii][iy -jj][iz -kk ];
                            Bzl += weight[ii][jj][kk]*field->Bzn[ix- ii][iy -jj][iz -kk ];
                        }
                
                // end interpolation
                omdtsq = qomdt2*qomdt2*(Bxl*Bxl+Byl*Byl+Bzl*Bzl);
                denom = 1.0/(1.0 + omdtsq);
                // solve the position equation
                ut= part->u[i] + qomdt2*Exl;
                vt= part->v[i] + qomdt2*Eyl;
                wt= part->w[i] + qomdt2*Ezl;
                udotb = ut*Bxl + vt*Byl + wt*Bzl;
                // solve the velocity equation
                uptilde = (ut+qomdt2*(vt*Bzl -wt*Byl + qomdt2*udotb*Bxl))*denom;
                vptilde = (vt+qomdt2*(wt*Bxl -ut*Bzl + qomdt2*udotb*Byl))*denom;
                wptilde = (wt+qomdt2*(ut*Byl -vt*Bxl + qomdt2*udotb*Bzl))*denom;
                // update position
                part->x[i] = xptilde + uptilde*dto2;
                part->y[i] = yptilde + vptilde*dto2;
                part->z[i] = zptilde + wptilde*dto2;
                
                
            } // end of iteration
            // update the final position and velocity
            part->u[i]= 2.0*uptilde - part->u[i];
            part->v[i]= 2.0*vptilde - part->v[i];
            part->w[i]= 2.0*wptilde - part->w[i];
            part->x[i] = xptilde + uptilde*dt_sub_cycling;
            part->y[i] = yptilde + vptilde*dt_sub_cycling;
            part->z[i] = zptilde + wptilde*dt_sub_cycling;
            
            
            //////////
            //////////
            ////////// BC
                                        
            // X-DIRECTION: BC particles
            if (part->x[i] > grd->Lx){
                if (param->PERIODICX==true){ // PERIODIC
                    part->x[i] = part->x[i] - grd->Lx;
                } else { // REFLECTING BC
                    part->u[i] = -part->u[i];
                    part->x[i] = 2*grd->Lx - part->x[i];
                }
            }
                                                                        
            if (part->x[i] < 0){
                if (param->PERIODICX==true){ // PERIODIC
                   part->x[i] = part->x[i] + grd->Lx;
                } else { // REFLECTING BC
                    part->u[i] = -part->u[i];
                    part->x[i] = -part->x[i];
                }
            }
                
            
            // Y-DIRECTION: BC particles
            if (part->y[i] > grd->Ly){
                if (param->PERIODICY==true){ // PERIODIC
                    part->y[i] = part->y[i] - grd->Ly;
                } else { // REFLECTING BC
                    part->v[i] = -part->v[i];
                    part->y[i] = 2*grd->Ly - part->y[i];
                }
            }
                                                                        
            if (part->y[i] < 0){
                if (param->PERIODICY==true){ // PERIODIC
                    part->y[i] = part->y[i] + grd->Ly;
                } else { // REFLECTING BC
                    part->v[i] = -part->v[i];
                    part->y[i] = -part->y[i];
                }
            }
                                                                        
            // Z-DIRECTION: BC particles
            if (part->z[i] > grd->Lz){
                if (param->PERIODICZ==true){ // PERIODIC
                    part->z[i] = part->z[i] - grd->Lz;
                } else { // REFLECTING BC
                    part->w[i] = -part->w[i];
                    part->z[i] = 2*grd->Lz - part->z[i];
                }
            }
                                                                        
            if (part->z[i] < 0){
                if (param->PERIODICZ==true){ // PERIODIC
                    part->z[i] = part->z[i] + grd->Lz;
                } else { // REFLECTING BC
                    part->w[i] = -part->w[i];
                    part->z[i] = -part->z[i];
                }
            }
                                                                        
            
            
        }  // end of subcycling
    } // end of one particle
                                                                        
    return(0); // exit succcesfully
} // end of the mover


__global__ void move_particles(particles* devicePart, EMfield* deviceField, grid* deviceGrd, parameters* deviceParam) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= devicePart->nop) return;

    // auxiliary variables
    FPpart dt_sub_cycling = (FPpart) deviceParam->dt/((double) devicePart->n_sub_cycles);
    FPpart dto2 = .5*dt_sub_cycling, qomdt2 = devicePart->qom*dto2/deviceParam->c;
    FPpart omdtsq, denom, ut, vt, wt, udotb;
    
    // local (to the particle) electric and magnetic field
    FPfield Exl=0.0, Eyl=0.0, Ezl=0.0, Bxl=0.0, Byl=0.0, Bzl=0.0;
    
    // interpolation densities
    int ix,iy,iz;
    FPfield weight[2][2][2];
    FPfield xi[2], eta[2], zeta[2];
    
    // intermediate particle position and velocity
    FPpart xptilde, yptilde, zptilde, uptilde, vptilde, wptilde;
    
    // start subcycling
    for (int i_sub=0; i_sub < devicePart->n_sub_cycles; i_sub++){
        // move each particle with new fields
        //for (int i=0; i < devicePart->nop; i++){
            xptilde = devicePart->x[idx];
            yptilde = devicePart->y[idx];
            zptilde = devicePart->z[idx];
            // calculate the average velocity iteratively
            for(int innter=0; innter < devicePart->NiterMover; innter++){
                // interpolation G-->P
                ix = 2 +  int((devicePart->x[idx] - deviceGrd->xStart)*deviceGrd->invdx);
                iy = 2 +  int((devicePart->y[idx] - deviceGrd->yStart)*deviceGrd->invdy);
                iz = 2 +  int((devicePart->z[idx] - deviceGrd->zStart)*deviceGrd->invdz);
                
                // calculate weights
                xi[0]   = devicePart->x[idx] - deviceGrd->XN_flat[get_idx(ix - 1, iy, iz, deviceGrd->nyn, deviceGrd->nzn)];
                eta[0]  = devicePart->y[idx] - deviceGrd->YN_flat[get_idx(ix, iy - 1, iz, deviceGrd->nyn, deviceGrd->nzn)];
                zeta[0] = devicePart->z[idx] - deviceGrd->ZN_flat[get_idx(ix, iy, iz - 1, deviceGrd->nyn, deviceGrd->nzn)];
                xi[1]   = deviceGrd->XN_flat[get_idx(ix, iy, iz, deviceGrd->nyn, deviceGrd->nzn)] - devicePart->x[idx];
                eta[1]  = deviceGrd->YN_flat[get_idx(ix, iy, iz, deviceGrd->nyn, deviceGrd->nzn)] - devicePart->y[idx];
                zeta[1] = deviceGrd->ZN_flat[get_idx(ix, iy, iz, deviceGrd->nyn, deviceGrd->nzn)] - devicePart->z[idx];
                for (int ii = 0; ii < 2; ii++)
                    for (int jj = 0; jj < 2; jj++)
                        for (int kk = 0; kk < 2; kk++)
                            weight[ii][jj][kk] = xi[ii] * eta[jj] * zeta[kk] * deviceGrd->invVOL;
                
                // set to zero local electric and magnetic field
                Exl=0.0, Eyl = 0.0, Ezl = 0.0, Bxl = 0.0, Byl = 0.0, Bzl = 0.0;
                
                for (int ii=0; ii < 2; ii++)
                    for (int jj=0; jj < 2; jj++)
                        for(int kk=0; kk < 2; kk++){
                            Exl += weight[ii][jj][kk]*deviceField->Ex_flat[get_idx(ix-ii, iy-jj, iz-kk, deviceGrd->nyn, deviceGrd->nzn)];
                            Eyl += weight[ii][jj][kk]*deviceField->Ey_flat[get_idx(ix-ii, iy-jj, iz-kk, deviceGrd->nyn, deviceGrd->nzn)];
                            Ezl += weight[ii][jj][kk]*deviceField->Ez_flat[get_idx(ix-ii, iy-jj, iz-kk, deviceGrd->nyn, deviceGrd->nzn)];
                            Bxl += weight[ii][jj][kk]*deviceField->Bxn_flat[get_idx(ix-ii, iy-jj, iz-kk, deviceGrd->nyn, deviceGrd->nzn)];
                            Byl += weight[ii][jj][kk]*deviceField->Byn_flat[get_idx(ix-ii, iy-jj, iz-kk, deviceGrd->nyn, deviceGrd->nzn)];
                            Bzl += weight[ii][jj][kk]*deviceField->Bzn_flat[get_idx(ix-ii, iy-jj, iz-kk, deviceGrd->nyn, deviceGrd->nzn)];
                        }
                
                // end interpolation
                omdtsq = qomdt2*qomdt2*(Bxl*Bxl+Byl*Byl+Bzl*Bzl);
                denom = 1.0/(1.0 + omdtsq);
                // solve the position equation
                ut= devicePart->u[idx] + qomdt2*Exl;
                vt= devicePart->v[idx] + qomdt2*Eyl;
                wt= devicePart->w[idx] + qomdt2*Ezl;
                udotb = ut*Bxl + vt*Byl + wt*Bzl;
                // solve the velocity equation
                uptilde = (ut+qomdt2*(vt*Bzl -wt*Byl + qomdt2*udotb*Bxl))*denom;
                vptilde = (vt+qomdt2*(wt*Bxl -ut*Bzl + qomdt2*udotb*Byl))*denom;
                wptilde = (wt+qomdt2*(ut*Byl -vt*Bxl + qomdt2*udotb*Bzl))*denom;
                // update position
                devicePart->x[idx] = xptilde + uptilde*dto2;
                devicePart->y[idx] = yptilde + vptilde*dto2;
                devicePart->z[idx] = zptilde + wptilde*dto2;            
                
            } // end of iteration
            // update the final position and velocity
            devicePart->u[idx]= 2.0*uptilde - devicePart->u[idx];
            devicePart->v[idx]= 2.0*vptilde - devicePart->v[idx];
            devicePart->w[idx]= 2.0*wptilde - devicePart->w[idx];
            devicePart->x[idx] = xptilde + uptilde*dt_sub_cycling;
            devicePart->y[idx] = yptilde + vptilde*dt_sub_cycling;
            devicePart->z[idx] = zptilde + wptilde*dt_sub_cycling;
            
            //////////
            //////////
            ////////// BC
                                        
            // X-DIRECTION: BC particles
            if (devicePart->x[idx] > deviceGrd->Lx){
                if (deviceParam->PERIODICX==true){ // PERIODIC
                    devicePart->x[idx] = devicePart->x[idx] - deviceGrd->Lx;
                } else { // REFLECTING BC
                    devicePart->u[idx] = -devicePart->u[idx];
                    devicePart->x[idx] = 2*deviceGrd->Lx - devicePart->x[idx];
                }
            }
                                                                        
            if (devicePart->x[idx] < 0){
                if (deviceParam->PERIODICX==true){ // PERIODIC
                   devicePart->x[idx] = devicePart->x[idx] + deviceGrd->Lx;
                } else { // REFLECTING BC
                    devicePart->u[idx] = -devicePart->u[idx];
                    devicePart->x[idx] = -devicePart->x[idx];
                }
            }
            
            // Y-DIRECTION: BC particles
            if (devicePart->y[idx] > deviceGrd->Ly){
                if (deviceParam->PERIODICY==true){ // PERIODIC
                    devicePart->y[idx] = devicePart->y[idx] - deviceGrd->Ly;
                } else { // REFLECTING BC
                    devicePart->v[idx] = -devicePart->v[idx];
                    devicePart->y[idx] = 2*deviceGrd->Ly - devicePart->y[idx];
                }
            }
                                                                        
            if (devicePart->y[idx] < 0){
                if (deviceParam->PERIODICY==true){ // PERIODIC
                    devicePart->y[idx] = devicePart->y[idx] + deviceGrd->Ly;
                } else { // REFLECTING BC
                    devicePart->v[idx] = -devicePart->v[idx];
                    devicePart->y[idx] = -devicePart->y[idx];
                }
            }
                                                                        
            // Z-DIRECTION: BC particles
            if (devicePart->z[idx] > deviceGrd->Lz){
                if (deviceParam->PERIODICZ==true){ // PERIODIC
                    devicePart->z[idx] = devicePart->z[idx] - deviceGrd->Lz;
                } else { // REFLECTING BC
                    devicePart->w[idx] = -devicePart->w[idx];
                    devicePart->z[idx] = 2*deviceGrd->Lz - devicePart->z[idx];
                }
            }
                                                                        
            if (devicePart->z[idx] < 0){
                if (deviceParam->PERIODICZ==true){ // PERIODIC
                    devicePart->z[idx] = devicePart->z[idx] + deviceGrd->Lz;
                } else { // REFLECTING BC
                    devicePart->w[idx] = -devicePart->w[idx];
                    devicePart->z[idx] = -devicePart->z[idx];
                }
            }
                                                                        
        //}  // end of subcycling
    } // end of one particle
}


/** particle mover using GPU*/
int mover_PC_gpu(struct particles* part, struct EMfield* field, struct grid* grd, struct parameters* param)
{
    //@@ Required variables on GPU
    particles* devicePart;
    EMfield* deviceField;
    grid* deviceGrd;
    parameters* deviceParam;

    //@@ Insert code below to allocate GPU memory here
    cudaMalloc(&devicePart, sizeof(particles));
    cudaMalloc(&deviceField, sizeof(EMfield));
    cudaMalloc(&deviceGrd, sizeof(grid));
    cudaMalloc(&deviceParam, sizeof(parameters));

    //@@ Insert code to Copy memory to the GPU here
    // Copy part struct. q array does not need to be copied because it is not used in mover_PC()
    cudaMemcpy(devicePart, part, sizeof(particles), cudaMemcpyHostToDevice);
    cudaMemcpy(deviceField, field, sizeof(EMfield), cudaMemcpyHostToDevice);
    cudaMemcpy(deviceGrd, grd, sizeof(grid), cudaMemcpyHostToDevice);
    cudaMemcpy(deviceParam, param, sizeof(parameters), cudaMemcpyHostToDevice);

    // Pointers stored in structs still point to the memory in host.
    // Hence, it is required to manually copy the arrays (https://stackoverflow.com/questions/31598021/cuda-cudamemcpy-struct-of-arrays).

    // Copy arrays of part struct
    FPpart *devicePart_x, *devicePart_y, *devicePart_z, *devicePart_u, *devicePart_v, *devicePart_w;
    cudaMalloc(&devicePart_x, part->npmax * sizeof(FPpart));
    cudaMalloc(&devicePart_y, part->npmax * sizeof(FPpart));
    cudaMalloc(&devicePart_z, part->npmax * sizeof(FPpart));
    cudaMalloc(&devicePart_u, part->npmax * sizeof(FPpart));
    cudaMalloc(&devicePart_v, part->npmax * sizeof(FPpart));
    cudaMalloc(&devicePart_w, part->npmax * sizeof(FPpart));

    cudaMemcpy(devicePart_x, part->x, part->npmax * sizeof(FPpart), cudaMemcpyHostToDevice);
    cudaMemcpy(devicePart_y, part->y, part->npmax * sizeof(FPpart), cudaMemcpyHostToDevice);
    cudaMemcpy(devicePart_z, part->z, part->npmax * sizeof(FPpart), cudaMemcpyHostToDevice);
    cudaMemcpy(devicePart_u, part->u, part->npmax * sizeof(FPpart), cudaMemcpyHostToDevice);
    cudaMemcpy(devicePart_v, part->v, part->npmax * sizeof(FPpart), cudaMemcpyHostToDevice);
    cudaMemcpy(devicePart_w, part->w, part->npmax * sizeof(FPpart), cudaMemcpyHostToDevice);

    // Binding pointers with devicePart struct
    cudaMemcpy(&(devicePart->x), &devicePart_x, sizeof(FPpart*), cudaMemcpyHostToDevice);
    cudaMemcpy(&(devicePart->y), &devicePart_y, sizeof(FPpart*), cudaMemcpyHostToDevice);
    cudaMemcpy(&(devicePart->z), &devicePart_z, sizeof(FPpart*), cudaMemcpyHostToDevice);
    cudaMemcpy(&(devicePart->u), &devicePart_u, sizeof(FPpart*), cudaMemcpyHostToDevice);
    cudaMemcpy(&(devicePart->v), &devicePart_v, sizeof(FPpart*), cudaMemcpyHostToDevice);
    cudaMemcpy(&(devicePart->w), &devicePart_w, sizeof(FPpart*), cudaMemcpyHostToDevice);

    // Copy arrays of field struct
    FPfield *deviceField_Ex_flat, *deviceField_Ey_flat, *deviceField_Ez_flat, *deviceField_Bxn_flat, *deviceField_Byn_flat, *deviceField_Bzn_flat;
    cudaMalloc(&deviceField_Ex_flat, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield));
    cudaMalloc(&deviceField_Ey_flat, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield));
    cudaMalloc(&deviceField_Ez_flat, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield));
    cudaMalloc(&deviceField_Bxn_flat, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield));
    cudaMalloc(&deviceField_Byn_flat, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield));
    cudaMalloc(&deviceField_Bzn_flat, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield));

    cudaMemcpy(deviceField_Ex_flat, field->Ex_flat, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield), cudaMemcpyHostToDevice);
    cudaMemcpy(deviceField_Ey_flat, field->Ey_flat, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield), cudaMemcpyHostToDevice);
    cudaMemcpy(deviceField_Ez_flat, field->Ez_flat, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield), cudaMemcpyHostToDevice);
    cudaMemcpy(deviceField_Bxn_flat, field->Bxn_flat, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield), cudaMemcpyHostToDevice);
    cudaMemcpy(deviceField_Byn_flat, field->Byn_flat, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield), cudaMemcpyHostToDevice);
    cudaMemcpy(deviceField_Bzn_flat, field->Bzn_flat, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield), cudaMemcpyHostToDevice);

    // Binding pointers with devicePart struct
    cudaMemcpy(&(deviceField->Ex_flat), &deviceField_Ex_flat, sizeof(FPfield*), cudaMemcpyHostToDevice);
    cudaMemcpy(&(deviceField->Ey_flat), &deviceField_Ey_flat, sizeof(FPfield*), cudaMemcpyHostToDevice);
    cudaMemcpy(&(deviceField->Ez_flat), &deviceField_Ez_flat, sizeof(FPfield*), cudaMemcpyHostToDevice);
    cudaMemcpy(&(deviceField->Bxn_flat), &deviceField_Bxn_flat, sizeof(FPfield*), cudaMemcpyHostToDevice);
    cudaMemcpy(&(deviceField->Byn_flat), &deviceField_Byn_flat, sizeof(FPfield*), cudaMemcpyHostToDevice);
    cudaMemcpy(&(deviceField->Bzn_flat), &deviceField_Bzn_flat, sizeof(FPfield*), cudaMemcpyHostToDevice);

    // Copy arrays of grd struct
    FPfield *deviceGrd_XN_flat, *deviceGrd_YN_flat, *deviceGrd_ZN_flat;
    cudaMalloc(&deviceGrd_XN_flat, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield));
    cudaMalloc(&deviceGrd_YN_flat, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield));
    cudaMalloc(&deviceGrd_ZN_flat, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield));

    cudaMemcpy(deviceGrd_XN_flat, grd->XN_flat, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield), cudaMemcpyHostToDevice);
    cudaMemcpy(deviceGrd_YN_flat, grd->YN_flat, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield), cudaMemcpyHostToDevice);
    cudaMemcpy(deviceGrd_ZN_flat, grd->ZN_flat, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield), cudaMemcpyHostToDevice);

    // Binding pointers with devicePart struct
    cudaMemcpy(&(deviceGrd->XN_flat), &deviceGrd_XN_flat, sizeof(FPfield*), cudaMemcpyHostToDevice);
    cudaMemcpy(&(deviceGrd->YN_flat), &deviceGrd_YN_flat, sizeof(FPfield*), cudaMemcpyHostToDevice);
    cudaMemcpy(&(deviceGrd->ZN_flat), &deviceGrd_ZN_flat, sizeof(FPfield*), cudaMemcpyHostToDevice);

    //@@ Initialize the grid and block dimensions here
    int Db_h = 64;
    int Dg_h = (part->nop + Db_h - 1) / Db_h;

    // print species and subcycling
    std::cout << "***  MOVER with SUBCYCLYING "<< param->n_sub_cycles << " - species " << part->species_ID << " ***" << std::endl;

    //@@ Launch the GPU Kernel here
    move_particles<<<Dg_h, Db_h>>>(devicePart, deviceField, deviceGrd, deviceParam);

    //@@ Copy the GPU memory back to the CPU here (be careful because the struct contain arrays which need to be copied separately)
    // Copy part struct
    cudaMemcpy(part, devicePart, sizeof(particles), cudaMemcpyDeviceToHost); // pointers to arrays in struct still point to device memory

    FPpart *hostPart_x, *hostPart_y, *hostPart_z, *hostPart_u, *hostPart_v, *hostPart_w;
    hostPart_x = (FPpart*) malloc(part->npmax * sizeof(FPpart));
    hostPart_y= (FPpart*) malloc(part->npmax * sizeof(FPpart));
    hostPart_z = (FPpart*) malloc(part->npmax * sizeof(FPpart));
    hostPart_u = (FPpart*) malloc(part->npmax * sizeof(FPpart));
    hostPart_v = (FPpart*) malloc(part->npmax * sizeof(FPpart));
    hostPart_w = (FPpart*) malloc(part->npmax * sizeof(FPpart));

    cudaMemcpy(hostPart_x, devicePart_x, part->npmax * sizeof(FPpart), cudaMemcpyDeviceToHost);
    cudaMemcpy(hostPart_y, devicePart_y, part->npmax * sizeof(FPpart), cudaMemcpyDeviceToHost);
    cudaMemcpy(hostPart_z, devicePart_z, part->npmax * sizeof(FPpart), cudaMemcpyDeviceToHost);
    cudaMemcpy(hostPart_u, devicePart_u, part->npmax * sizeof(FPpart), cudaMemcpyDeviceToHost);
    cudaMemcpy(hostPart_v, devicePart_v, part->npmax * sizeof(FPpart), cudaMemcpyDeviceToHost);
    cudaMemcpy(hostPart_w, devicePart_w, part->npmax * sizeof(FPpart), cudaMemcpyDeviceToHost);

    // Binding pointers with part struct
    part->x = hostPart_x;
    part->y = hostPart_y;
    part->z = hostPart_z;
    part->u = hostPart_u;
    part->v = hostPart_v;
    part->w = hostPart_w;

    //@@ Free the GPU memory here
    cudaFree(devicePart);
    cudaFree(deviceField);
    cudaFree(deviceGrd);
    cudaFree(deviceParam);                                                               

    return(0); // exit succcesfully
} // end of the mover


/** Interpolation Particle --> Grid: This is for species */
void interpP2G(struct particles* part, struct interpDensSpecies* ids, struct grid* grd)
{
    
    // arrays needed for interpolation
    FPpart weight[2][2][2];
    FPpart temp[2][2][2];
    FPpart xi[2], eta[2], zeta[2];
    
    // index of the cell
    int ix, iy, iz;
    
    
    for (register long long i = 0; i < part->nop; i++) {
        
        // determine cell: can we change to int()? is it faster?
        ix = 2 + int (floor((part->x[i] - grd->xStart) * grd->invdx));
        iy = 2 + int (floor((part->y[i] - grd->yStart) * grd->invdy));
        iz = 2 + int (floor((part->z[i] - grd->zStart) * grd->invdz));

        // distances from node
        xi[0]   = part->x[i] - grd->XN[ix - 1][iy][iz];
        eta[0]  = part->y[i] - grd->YN[ix][iy - 1][iz];
        zeta[0] = part->z[i] - grd->ZN[ix][iy][iz - 1];
        xi[1]   = grd->XN[ix][iy][iz] - part->x[i];
        eta[1]  = grd->YN[ix][iy][iz] - part->y[i];
        zeta[1] = grd->ZN[ix][iy][iz] - part->z[i];
        
        // calculate the weights for different nodes
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    weight[ii][jj][kk] = part->q[i] * xi[ii] * eta[jj] * zeta[kk] * grd->invVOL;
        
        //////////////////////////
        // add charge density
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    ids->rhon[ix - ii][iy - jj][iz - kk] += weight[ii][jj][kk] * grd->invVOL;
        
        
        ////////////////////////////
        // add current density - Jx
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    temp[ii][jj][kk] = part->u[i] * weight[ii][jj][kk];
        
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    ids->Jx[ix - ii][iy - jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;
        
        
        ////////////////////////////
        // add current density - Jy
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    temp[ii][jj][kk] = part->v[i] * weight[ii][jj][kk];
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    ids->Jy[ix - ii][iy - jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;
        
        
        
        ////////////////////////////
        // add current density - Jz
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    temp[ii][jj][kk] = part->w[i] * weight[ii][jj][kk];
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    ids->Jz[ix - ii][iy - jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;
        
        
        ////////////////////////////
        // add pressure pxx
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    temp[ii][jj][kk] = part->u[i] * part->u[i] * weight[ii][jj][kk];
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    ids->pxx[ix - ii][iy - jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;
        
        
        ////////////////////////////
        // add pressure pxy
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    temp[ii][jj][kk] = part->u[i] * part->v[i] * weight[ii][jj][kk];
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    ids->pxy[ix - ii][iy - jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;
        
        
        
        /////////////////////////////
        // add pressure pxz
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    temp[ii][jj][kk] = part->u[i] * part->w[i] * weight[ii][jj][kk];
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    ids->pxz[ix - ii][iy - jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;
        
        
        /////////////////////////////
        // add pressure pyy
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    temp[ii][jj][kk] = part->v[i] * part->v[i] * weight[ii][jj][kk];
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    ids->pyy[ix - ii][iy - jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;
        
        
        /////////////////////////////
        // add pressure pyz
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    temp[ii][jj][kk] = part->v[i] * part->w[i] * weight[ii][jj][kk];
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    ids->pyz[ix - ii][iy - jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;
        
        
        /////////////////////////////
        // add pressure pzz
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    temp[ii][jj][kk] = part->w[i] * part->w[i] * weight[ii][jj][kk];
        for (int ii=0; ii < 2; ii++)
            for (int jj=0; jj < 2; jj++)
                for(int kk=0; kk < 2; kk++)
                    ids->pzz[ix -ii][iy -jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;
    
    }
   
}