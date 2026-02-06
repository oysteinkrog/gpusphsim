#ifndef __K_SimpleSPH_Integrate_cu__
#define __K_SimpleSPH_Integrate_cu__

#include "K_Coloring.inl"
#include "K_Boundaries_Terrain.inl"
#include "K_Boundaries_Walls.inl"

template<SPHColoringSource coloringSource, ColoringGradient coloringGradient, bool gridWallCollisions, bool terrainCollisions>
__global__ void K_Integrate(int				numParticles,
							float			delta_time,
							bool			progress,
							GridData		dGridData,
							SimpleSPHData	dParticleData,
							SimpleSPHData	dParticleDataSorted,
							float3			fluidWorldPosition,
							TerrainData		dTerrainData
							)
{
	// standard multiplication is as fast as __umul24 on sm_20+
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= numParticles) return;

	// Sorted position is in simulation space; un-scale to world space for boundaries and writeback
	float3 pos			= make_float3(FETCH_NOTEX(dParticleDataSorted, position, index)) * cPrecalcParams.inv_scale_to_simulation;
	float3 vel			= make_float3(FETCH_NOTEX(dParticleDataSorted, velocity, index));
	float3 vel_eval		= make_float3(FETCH_NOTEX(dParticleDataSorted, veleval, index));

	float3 sph_force	= make_float3(FETCH_NOTEX(dParticleDataSorted, sph_force, index));
	// Recompute pressure from density for coloring (pressure buffer eliminated)
	float  sph_density	= FETCH_NOTEX(dParticleDataSorted, density, index);
	float  sph_pressure	= cFluidParams.rest_pressure + cFluidParams.gas_stiffness * (sph_density - cFluidParams.rest_density);

	float3 external_force = make_float3(0,0,0);
	// add gravity
	external_force.y -= 9.8f;


	// add no-penetration force due to terrain
	if(terrainCollisions)
		external_force += calculateTerrainNoPenetrationForce(
			pos, vel_eval,
			fluidWorldPosition, dTerrainData,
			cFluidParams.boundary_distance,
			cFluidParams.boundary_stiffness,
			cFluidParams.boundary_dampening,
			cFluidParams.scale_to_simulation);

	// add no-slip force due to terrain..
	if(terrainCollisions)
		external_force += calculateTerrainFrictionForce(
			pos, vel_eval, sph_force+external_force,
			fluidWorldPosition, dTerrainData,
			cFluidParams.boundary_distance,
			cFluidParams.friction_kinetic/delta_time,
			cFluidParams.friction_static_limit,
			cFluidParams.scale_to_simulation);


	// add no-penetration force due to "walls"
	if(gridWallCollisions)
		external_force += calculateWallsNoPenetrationForce(
			pos, vel_eval,
			cGridParams.grid_min,
			cGridParams.grid_max,
			cFluidParams.boundary_distance,
			cFluidParams.boundary_stiffness,
			cFluidParams.boundary_dampening,
			cFluidParams.scale_to_simulation);

	// add no-slip force due to "walls"
	if(gridWallCollisions)
		external_force += calculateWallsNoSlipForce(
			pos, vel_eval, sph_force + external_force,
			cGridParams.grid_min,
			cGridParams.grid_max,
			cFluidParams.boundary_distance,
			cFluidParams.friction_kinetic/delta_time,
			cFluidParams.friction_static_limit,
			cFluidParams.scale_to_simulation);

	float3 force = sph_force + external_force;

	// limit velocity — compare squared magnitudes to avoid sqrt in the common case
	float speed_sq = dot(force, force);
	if (speed_sq > cPrecalcParams.velocity_limit_sq) {
		float speed = sqrtf(speed_sq);
		force *= cFluidParams.velocity_limit / speed;
	}

	// Leapfrog integration
	// v(t+1/2) = v(t-1/2) + a(t) dt
	float3 vnext = vel + force * delta_time;
	// v(t+1) = [v(t-1/2) + v(t+1/2)] * 0.5
	vel_eval = (vel + vnext) * 0.5;
	vel = vnext;

	// update position of particle (pos is in world space)
	pos += vnext * (delta_time * cPrecalcParams.inv_scale_to_simulation);

	if(progress)
	{
		uint originalIndex = dGridData.sort_indexes[index];

		// writeback to unsorted buffer (world space)
		dParticleData.position[originalIndex]	= make_vec(pos);
		dParticleData.velocity[originalIndex]	= make_vec(vel);
		dParticleData.veleval[originalIndex]	= make_vec(vel_eval);

		float3 color = CalculateColor(coloringGradient, coloringSource, vnext, sph_pressure, sph_force);
		dParticleData.color[originalIndex]	= make_float4(color, 1);
	}
}

#endif
