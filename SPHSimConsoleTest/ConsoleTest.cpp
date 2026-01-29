
#include <SimulationSystem.h>

#include <iostream>
#include <iomanip>
#include <numeric>
#include <limits>
#include <conio.h> 

using namespace std;
using namespace SimLib;
using namespace ocu;

#define MEASURE_KERNEL_TIMING false

void pause()
{
	_getch(); 

// 	cin.clear();
// 	cin.ignore(std::numeric_limits<streamsize>::max());   //
// 	cout << "Press Enter to continue . . .\n";
// 	cin.ignore(std::numeric_limits<streamsize>::max(),'\n');
}

void testFluidSimLive(SimLib::SimCudaHelper* simCudaHelper)
{
	SimulationSystem *system = new SimulationSystem(true, simCudaHelper, MEASURE_KERNEL_TIMING);
	system->Init();

	system->GetSettings()->SetValue("Timestep", 0.002);
	system->GetSettings()->SetValue("Particles Number", 128*1024);
	system->GetSettings()->SetValue("Grid World Size",  1024);
	system->GetSettings()->SetValue("Simulation Scale",  0.0005);
	system->GetSettings()->SetValue("Rest Density",  1000);
	system->GetSettings()->SetValue("Rest Pressure", 0);
	system->GetSettings()->SetValue("Ideal Gas Constant",  1.5);
	system->GetSettings()->SetValue("Viscosity",  1);
	system->GetSettings()->SetValue("Boundary Stiffness", 30);
	system->GetSettings()->SetValue("Boundary Dampening", 30);

	system->GetSettings()->Print();
	cout << "\n";
	//system->SetPrintTiming(true);
	system->SetScene(6);

	cout << "\nFPS:\n";
	cout << setw(15) << "Current";
	cout << setw(15) << "Avg (10)";
	cout << setw(15) << "Avg Total";
	cout << "\n";

	double totalavg=0;
	double fpshistory[100] = {0};
	GPUTimer *timer = new GPUTimer();

	 //CPUTimer *totalTimer = new CPUTimer();
	 //totalTimer->start();

	int ITERATIONS = 200;
	for(int i = 0; i < ITERATIONS; i++)
	{
		timer->start();
		system->Simulate(true, true);
		timer->stop();

		// calc fps
		double fps = 1000.0/timer->elapsed_ms();

		// calc running average of full history
		totalavg = (fps+i*totalavg)/(i+1);

		// store fps in history
		fpshistory[i%100] = fps;

		// get average of fps history
		double avg=0; for(int j=0;j<100;j++) fpshistory[j]==0? avg += fps : avg += fpshistory[j]; avg /= 100.0;

		cout << fixed;
		cout << setw(15) << setprecision(1) << fps ;
		cout << setw(15) << setprecision(1) << avg ;
		cout << setw(15) << setprecision(1) <<  totalavg;
		cout << "\r";
	}
	cudaDeviceSynchronize();

	// 	totalTimer->stop();
	// 	double totalTime = totalTimer->elapsed_ms();
	// 	cout << "Total ms: " << totalTime << "\n";
	// 	cout << "Avg ms/frame: " << totalTime/ITERATIONS << "\n";
	// 	cout << "Avg fps: " << 1000.0/(totalTime/ITERATIONS) << "\n";

	pause();
}

void testFluidSim(SimLib::SimCudaHelper* simCudaHelper)
{
	SimulationSystem *system = new SimulationSystem(true, simCudaHelper, MEASURE_KERNEL_TIMING);
	system->Init();

	system->GetSettings()->SetValue("Timestep", 0.002);
	system->GetSettings()->SetValue("Particles Number", 32*1024);
	system->GetSettings()->SetValue("Grid World Size",  1024);
	system->GetSettings()->SetValue("Simulation Scale",  0.002);
	system->GetSettings()->SetValue("Rest Density",  1000);
	system->GetSettings()->SetValue("Rest Pressure", 0);
	system->GetSettings()->SetValue("Ideal Gas Constant",  1.5);
	system->GetSettings()->SetValue("Viscosity",  1);
 	system->GetSettings()->SetValue("Boundary Stiffness", 20000);
 	system->GetSettings()->SetValue("Boundary Dampening", 256);

	system->GetSettings()->Print();
	cout << "\n";
	//system->SetPrintTiming(true);
	system->SetScene(6);

	GPUTimer *totalTimer = new GPUTimer();
	totalTimer->start();

	system->PrintMemoryUse();

	int ITERATIONS = 1000;
	for(int i = 0; i < ITERATIONS; i++)
	{
		system->Simulate(true, true);
	}
	cudaDeviceSynchronize();


	totalTimer->stop();
	double totalTime = totalTimer->elapsed_ms();
	cout << "Total ms: " << totalTime << "\n";
	cout << "Avg ms/frame: " << totalTime/ITERATIONS << "\n";
	cout << "Avg fps: " << 1000.0/(totalTime/ITERATIONS) << "\n";

	pause();
}

void testPerformanceScaling(SimLib::SimCudaHelper* simCudaHelper)
{
	SimulationSystem *system = new SimulationSystem(true, simCudaHelper, MEASURE_KERNEL_TIMING);
	system->Init();

	system->GetSettings()->SetValue("Timestep", 0.0005);
	system->GetSettings()->SetValue("Grid World Size",  1024);
	system->GetSettings()->SetValue("Simulation Scale",  0.0005);
	system->GetSettings()->SetValue("Rest Density",  1000);
	system->GetSettings()->SetValue("Rest Pressure", 0);
	system->GetSettings()->SetValue("Ideal Gas Constant",  1.5);
	system->GetSettings()->SetValue("Viscosity",  1);
	system->GetSettings()->SetValue("Boundary Stiffness",  20000);
	system->GetSettings()->SetValue("Boundary Dampening", 256);

	system->GetSettings()->Print();
	cout << "\n";

	uint startParticles = 1*1024;
	uint endParticles = 512*1024;

	int ITERATIONS = 1000;

	float psizes[100] = {0};
	float vals[100] = {0};

	uint DOUBLINGS = 0;
	for(uint numParticles = startParticles; numParticles<=endParticles; DOUBLINGS++,numParticles *= 2) 
	{
		psizes[DOUBLINGS] = numParticles;
		system->GetSettings()->SetValue("Particles Number", numParticles);

		//system->SetPrintTiming(true);
		system->SetScene(6);

	 	CPUTimer *totalTimer = new CPUTimer();
 		totalTimer->start();

		cout << "\n";
		for(int i = 0; i < ITERATIONS; i++)
		{
			system->Simulate(true, true);
			//cout << "\r" << i << "/" << ITERATIONS;
		}
		cudaDeviceSynchronize();

		totalTimer->stop();
		double totalTime = totalTimer->elapsed_ms();
		vals[DOUBLINGS] = totalTime;
		cout << "\nRESULTS: Total ms: " << totalTime << "\t";
		cout << "Avg ms/frame: " << totalTime/ITERATIONS << "\t";
		cout << "Avg fps: " << 1000.0/(totalTime/ITERATIONS) << "\n\n";

	}
	cout << "SUMMARY OF TIMES:\n";
	for(int i = 0; i<DOUBLINGS;i++)
		cout << setw(10) << psizes[i] << " ";
	cout << "\n";
	for(int i = 0; i<DOUBLINGS;i++)
		cout << setw(10) << 1000.0/(vals[i]/ITERATIONS) << ", ";

	pause();
}

void testKernel();

void testKernelPerformance(SimLib::SimCudaHelper* simCudaHelper, bool verify = false)
{
	const int WARMUP = 10;
	const int TIMED = 100;
	const uint particleCounts[] = {1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288};
	const int numCounts = sizeof(particleCounts) / sizeof(particleCounts[0]);

	SimulationSystem *system = new SimulationSystem(true, simCudaHelper, true);
	system->Init();

	// Configure simulation parameters
	system->GetSettings()->SetValue("Timestep", 0.0005);
	system->GetSettings()->SetValue("Grid World Size", 1024);
	system->GetSettings()->SetValue("Simulation Scale", 0.0005);
	system->GetSettings()->SetValue("Rest Density", 1000);
	system->GetSettings()->SetValue("Rest Pressure", 0);
	system->GetSettings()->SetValue("Ideal Gas Constant", 1.5);
	system->GetSettings()->SetValue("Viscosity", 1);
	system->GetSettings()->SetValue("Boundary Stiffness", 20000);
	system->GetSettings()->SetValue("Boundary Dampening", 256);

	// Print CSV header
	cout << "particles,hash_ms,sort_ms,build_ms,step1_ms,step2_ms,integrate_ms,total_ms,fps\n";

	for (int c = 0; c < numCounts; c++)
	{
		uint numParticles = particleCounts[c];
		system->GetSettings()->SetValue("Particles Number", numParticles);
		system->SetScene(6);

		// Warmup iterations (not timed)
		for (int i = 0; i < WARMUP; i++)
		{
			system->Simulate(true, true);
		}
		cudaDeviceSynchronize();

		// Accumulate timing
		double total_hash = 0, total_sort = 0, total_build = 0;
		double total_step1 = 0, total_step2 = 0, total_integrate = 0, total_total = 0;

		for (int i = 0; i < TIMED; i++)
		{
			system->Simulate(true, true);
			const SimLib::Sim::SimTimingResult& t = system->GetLastTimingResult();
			total_hash += t.hash_ms;
			total_sort += t.sort_ms;
			total_build += t.build_ms;
			total_step1 += t.step1_ms;
			total_step2 += t.step2_ms;
			total_integrate += t.integrate_ms;
			total_total += t.total_ms;
		}
		cudaDeviceSynchronize();

		// Compute averages
		double avg_hash = total_hash / TIMED;
		double avg_sort = total_sort / TIMED;
		double avg_build = total_build / TIMED;
		double avg_step1 = total_step1 / TIMED;
		double avg_step2 = total_step2 / TIMED;
		double avg_integrate = total_integrate / TIMED;
		double avg_total = total_total / TIMED;
		double fps = 1000.0 / avg_total;

		cout << fixed << setprecision(4);
		cout << numParticles << ","
			<< avg_hash << ","
			<< avg_sort << ","
			<< avg_build << ","
			<< avg_step1 << ","
			<< avg_step2 << ","
			<< avg_integrate << ","
			<< avg_total << ","
			<< setprecision(1) << fps << "\n";
	}

	delete system;
}

int main(int argc, char *argv[])
{
	SimLib::SimCudaHelper* simCudaHelper = new SimLib::SimCudaHelper();
	simCudaHelper->Initialize(0);

	//force the GPU to wake up
	//cudaEvent_t wakeGPU;
	//cutilSafeCall( cudaEventCreate( &wakeGPU) );
	//Sleep(1000);


	testKernelPerformance(simCudaHelper);
	//testPerformanceScaling(simCudaHelper);
	//testFluidSimLive(simCudaHelper);
	//testFluidSim(simCudaHelper);
	//testKernel();
	
}

