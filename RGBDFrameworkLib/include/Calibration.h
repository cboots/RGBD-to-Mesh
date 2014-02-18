#pragma once

namespace rgbd
{
	namespace framework
	{
		struct Intrinsics
		{
			float fx, fy, cx,  cy;
		public:
			Intrinsics(){}
			Intrinsics(float _fx, float _fy, float _cx, float _cy) 
				: fx(_fx), fy(_fy), cx(_cx), cy(_cy)	{}
		};

	}
}