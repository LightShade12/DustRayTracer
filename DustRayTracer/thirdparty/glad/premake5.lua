project "Glad"
    kind "StaticLib"
    language "C"
    staticruntime "off"
    
    targetname "%{prj.name}-%{cfg.buildcfg}-%{cfg.architecture}"
	targetdir ("bin/" .. outputdir .. "/")
	objdir ("bin-obj/" .. outputdir .. "/")

    files
    {
        "include/glad/glad.h",
        "include/KHR/khrplatform.h",
        "src/glad.c"
    }

    includedirs
    {
        "include"
    }
    
    filter "system:windows"
        systemversion "latest"
        cppdialect "C++17"

    filter "configurations:Debug"
        runtime "Debug"
        symbols "On"

    filter "configurations:Release"
        runtime "Release"
        optimize "Full"