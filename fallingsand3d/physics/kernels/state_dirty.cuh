/*
 * state_dirty.cuh -- packed_info bit 21: "reaction state dirty"
 * (bd-r5-review-n28.162).
 *
 * K_Reactions and K_SpawnGas change health, lifetime and mass in sorted
 * space for a small subset of particles. They set this bit on every
 * particle whose health, lifetime or mass they may have changed. K_Integrate
 * scatters those three arrays back to the unsorted arrays only when the bit
 * is set, and clears it in the packed_info it writes back. Every other
 * particle's unsorted copy already holds the value it was gathered from.
 *
 * The bit is a hint: setting it on a particle whose state did not change
 * only costs the three scattered stores. The PBF and DFSPH finalize kernels
 * scatter the three arrays unconditionally and carry the bit through
 * unchanged; every reader masks packed_info fields, so a set bit is harmless.
 *
 * common.cuh reserves bits [31:21] of packed_info; this is bit 21.
 */
#ifndef STATE_DIRTY_CUH
#define STATE_DIRTY_CUH

#define STATE_DIRTY_BIT       0x200000u
#define HAS_STATE_DIRTY(p)    (((p) >> 21) & 1u)
#define SET_STATE_DIRTY(p)    ((p) | STATE_DIRTY_BIT)
#define CLEAR_STATE_DIRTY(p)  ((p) & ~STATE_DIRTY_BIT)

#endif /* STATE_DIRTY_CUH */
