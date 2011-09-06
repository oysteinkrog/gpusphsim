
set inputfile=%1
set inputfilename=%inputfile:~0,-4%
set filetype=%inputfile:~-3%

convert -crop 512x512+512+0 %inputfile% %inputfilename%_FR.%filetype%
convert -crop 512x512+0+512 -rotate +90 %inputfile% %inputfilename%_LF.%filetype%
convert -crop 512x512+1024+512 -rotate -90 %inputfile% %inputfilename%_RT.%filetype%
convert -crop 512x512+512+512 %inputfile% %inputfilename%_DN.%filetype%
convert -crop 512x512+512+1024 -rotate 180 %inputfile% %inputfilename%_BK.%filetype%
convert -crop 512x512+512+1536 %inputfile% %inputfilename%_UP.%filetype%