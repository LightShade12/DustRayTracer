outputdir="/%{prj.name}-%{cfg.buildcfg}"

IncludeDir={}
IncludeDir["glad"]="VoxelToy/thirdparty/glad/include"
IncludeDir["stb_image"]="VoxelToy/thirdparty/stb_image"
IncludeDir["glm"]="VoxelToy/thirdparty/glm"
IncludeDir["glfw"]="VoxelToy/thirdparty/glfw_3.4/include"


workspace "VoxelToy"
	architecture "x64"
	configurations{
		"Debug",
		"Release"
	}

	
include "VoxelToy/thirdparty/glad"

project "VoxelToy"
	location "VoxelToy"
	kind "ConsoleApp"
	language "C++"
	
	targetname "%{prj.name}-%{cfg.buildcfg}-%{cfg.architecture}"
	targetdir ("bin" .. outputdir .. "/")
	objdir ("bin-obj" .. outputdir .. "/")
	
	files{
		"%{prj.name}/src/**.h",
		"%{prj.name}/src/**.cpp",
		"%{prj.name}/thirdparty/stb_image/**.h",
		"%{prj.name}/thirdparty/stb_image/**.cpp"
	}

	removefiles{}

	includedirs{
		"%{prj.name}/src",
		"%{IncludeDir.glad}",
		"%{IncludeDir.stb_image}",
		"%{IncludeDir.glm}",
		"%{IncludeDir.glfw}"
	}

	libdirs{
		"%{prj.name}/thirdparty/glfw_3.4/lib-vc2022/"
	}

	links{
		"glfw3.lib",
		"glad"
	}

	filter "system:windows"
		cppdialect "C++17"
		staticruntime "Off"
		systemversion "latest"

	filter "configurations:Release"
		optimize "On"
	
	filter "configurations:Debug"
		symbols "On"
