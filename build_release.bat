@echo off
set "CudaToolkitDir=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.5"
set "CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.5"
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
cd /d C:\WORK\gpusphsim
msbuild SPHSimOgreApp\SPHSimOgreApp.vcxproj /p:Configuration=Release /p:Platform=x64 /verbosity:minimal
