require("thirdparty/premake5-cuda/premake5-cuda")

--variables--------------
IncludeDir={}
IncludeDir["stb_image"]="DustRayTracer/thirdparty/stb_image"
IncludeDir["glfw"]="DustRayTracer/thirdparty/glfw/include"
IncludeDir["glad"]="DustRayTracer/thirdparty/glad/include"
IncludeDir["glm"]="DustRayTracer/thirdparty/glm"
IncludeDir["imgui"]="DustRayTracer/thirdparty/imgui"

outputdir="%{prj.name}-%{cfg.buildcfg}"
----------------------------------------------------

workspace "DustRayTracer"
	architecture "x64"
	configurations{
		"Debug",
		"Release"
	}

include "DustRayTracer/thirdparty/glad"
include "DustRayTracer/thirdparty/imgui"
include "DustRayTracer/thirdparty/glfw"

project "DustRayTracer"
	location "DustRayTracer"
	kind "ConsoleApp"
	language "C++"
	
	filter "system:windows"
		cppdialect "C++17"
		staticruntime "Off"
		systemversion "latest"
	
	targetname "%{prj.name}-%{cfg.buildcfg}-%{cfg.architecture}"
	targetdir ("bin/" .. outputdir .. "/")
	objdir ("bin-obj/" .. outputdir .. "/")
	

	files{
		"%{prj.name}/src/**.hpp",
		"%{prj.name}/src/**.cpp",
		"%{prj.name}/thirdparty/stb_image/**.h",
		"%{prj.name}/thirdparty/stb_image/**.cpp"
		}

	buildcustomizations "BuildCustomizations/CUDA 12.4"
	
	cudaFiles { "DustRayTracer/src/**.cu" }
	cudaKeep "Off" -- keep temporary output files
	cudaFastMath "Off"
	cudaRelocatableCode "On"
	cudaVerbosePTXAS "Off"
	cudaMaxRegCount "0"
	cudaIntDir "../bin-obj/cuda/%{cfg.buildcfg}-%{cfg.architecture}/"

	externalwarnings "Off" -- thrust gives a lot of warnings

	removefiles{
		"%{prj.name}/src/kernel.cu"
	}

	includedirs{
		"%{prj.name}/src",
		"%{IncludeDir.stb_image}",
		"%{IncludeDir.glfw}",
		"%{IncludeDir.glad}",
		"%{IncludeDir.glm}",
		"%{IncludeDir.imgui}"
	}


	links{
		"GLFW",
		"Glad",
		"ImGui"
	}
	
	cudaCompilerOptions {"-arch=compute_75", "-code=sm_75" ,"-t0", "--expt-relaxed-constexpr"} 

	filter "configurations:Release"
		symbols "Off"
		optimize "Full"
		runtime "Release"
		cudaFastMath "On"
		cudaGenLineInfo "On"

	filter "configurations:Debug"
		defines {"DEBUG"}
		symbols "On"
		optimize "Off"
		runtime "Debug"
		cudaLinkerOptions { "-g" }
