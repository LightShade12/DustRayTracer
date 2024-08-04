#pragma once

//TOOD: complete these guidelines
//all should be hpp
#include "Core/Renderer.hpp"//let client access to high level render api a.k.a PathTracerRenderer

#include "Core/Scene/HostScene.hpp"//let client parse and populate the scene description

#include "Core/Scene/HostCamera.hpp"//client gets interface and methods to interact with device camera; never directly touching its data

#include "Core/BVH/BVHBuilder.hpp"//should be handled by renderer, not client; build on renderer init

//Done= HostCamera