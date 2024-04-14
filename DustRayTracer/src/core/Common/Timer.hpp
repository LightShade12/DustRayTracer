#pragma once
/*
---------------------------------------------------------
* Timer fron Walnut Runtime: by Yan Chernikov
---------------------------------------------------------
*/

#include <iostream>
#include <string>
#include <chrono>

class Timer
{
public:
	Timer();
	

	void Reset();
	

	float Elapsed() const;
	

	float ElapsedMillis() const;

private:
	std::chrono::time_point<std::chrono::high_resolution_clock> m_Start;
};

class ScopedTimer
{
public:
	ScopedTimer(const std::string& name)
		: m_Name(name) {}
	~ScopedTimer()
	{
		float time = m_Timer.ElapsedMillis();
		std::cout << "[TIMER] " << m_Name << " - " << time << "ms\n";
	}
private:
	std::string m_Name;
	Timer m_Timer;
};
