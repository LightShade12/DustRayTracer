struct Color3
{
	float r, g, b;

	Color3() = default;
	Color3(float a) :r(a), g(a), b(a) {};
	Color3(float r, float g, float b) :r(r), g(g), b(b) {};

	float toScalar() const { return (r + g + b) / 3; }

	Color3 operator + (const Color3& other) {
		return Color3(this->r + other.r, this->g + other.g, this->b + other.b);
	}

	Color3 operator - (const Color3& other) {
		return Color3(this->r - other.r, this->g - other.g, this->b - other.b);
	}

	Color3 operator * (const Color3& other) {
		return Color3(this->r * other.r, this->g * other.g, this->b * other.b);
	}

	Color3 operator / (const Color3& other) {
		return Color3(this->r / other.r, this->g / other.g, this->b / other.b);
	}

	Color3 operator + (const float& other) {
		return Color3(this->r + other, this->g + other, this->b + other);
	}

	Color3 operator - (const float& other) {
		return Color3(this->r - other, this->g - other, this->b - other);
	}

	Color3 operator * (const float& other) {
		return Color3(this->r * other, this->g * other, this->b * other);
	}

	Color3 operator / (const float& other) {
		return Color3(this->r / other, this->g / other, this->b / other);
	}
};

void test() {
	Color3 fcol;
	fcol = { 1,1,1 };

	Color3 col = fcol - Color3(1);
}