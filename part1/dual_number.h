#ifndef DUAL_NUMBER_H
#define DUAL_NUMBER_H

#include <vector>
#include <cmath>
#include <algorithm>
#include <iostream>

class dual_number {
private:
    float value_; // primal part
    float dual_;  // dual part (derivative)

public:
    // Constructors
    dual_number() : value_(0.0f), dual_(0.0f) {}
    dual_number(float value) : value_(value), dual_(0.0f) {}
    dual_number(float value, float dual) : value_(value), dual_(dual) {}

    // Accessors
    float value() const { return value_; }
    float dual() const { return dual_; }

    // Basic arithmetic operations
    // Addition
    dual_number operator+(const dual_number& rhs) const {
        return dual_number(value_ + rhs.value_, dual_ + rhs.dual_);
    }

    // Subtraction
    dual_number operator-(const dual_number& rhs) const {
        return dual_number(value_ - rhs.value_, dual_ - rhs.dual_);
    }

    // Multiplication
    dual_number operator*(const dual_number& rhs) const {
        // (a + bε) * (c + dε) = a*c + (a*d + b*c)ε
        return dual_number(value_ * rhs.value_, 
                           value_ * rhs.dual_ + dual_ * rhs.value_);
    }

    // Division
    dual_number operator/(const dual_number& rhs) const {
        // (a + bε) / (c + dε) = a/c + (b*c - a*d)/(c*c)ε
        float val = value_ / rhs.value_;
        float du = (dual_ * rhs.value_ - value_ * rhs.dual_) / 
                   (rhs.value_ * rhs.value_);
        return dual_number(val, du);
    }

    // Unary negation
    dual_number operator-() const {
        return dual_number(-value_, -dual_);
    }

    // Assignment operators
    dual_number& operator+=(const dual_number& rhs) {
        value_ += rhs.value_;
        dual_ += rhs.dual_;
        return *this;
    }

    dual_number& operator-=(const dual_number& rhs) {
        value_ -= rhs.value_;
        dual_ -= rhs.dual_;
        return *this;
    }

    dual_number& operator*=(const dual_number& rhs) {
        float new_dual = value_ * rhs.dual_ + dual_ * rhs.value_;
        value_ *= rhs.value_;
        dual_ = new_dual;
        return *this;
    }

    dual_number& operator/=(const dual_number& rhs) {
        float new_dual = (dual_ * rhs.value_ - value_ * rhs.dual_) / 
                         (rhs.value_ * rhs.value_);
        value_ /= rhs.value_;
        dual_ = new_dual;
        return *this;
    }

    // Comparison operators
    bool operator==(const dual_number& rhs) const {
        return value_ == rhs.value_ && dual_ == rhs.dual_;
    }

    bool operator!=(const dual_number& rhs) const {
        return !(*this == rhs);
    }
};

// Elementary functions
inline dual_number sin(const dual_number& x) {
    return dual_number(std::sin(x.value()), std::cos(x.value()) * x.dual());
}

inline dual_number cos(const dual_number& x) {
    return dual_number(std::cos(x.value()), -std::sin(x.value()) * x.dual());
}

inline dual_number exp(const dual_number& x) {
    float exp_val = std::exp(x.value());
    return dual_number(exp_val, exp_val * x.dual());
}

inline dual_number ln(const dual_number& x) {
    return dual_number(std::log(x.value()), x.dual() / x.value());
}

inline dual_number relu(const dual_number& x) {
    if (x.value() > 0) {
        return dual_number(x.value(), x.dual());
    } else {
        return dual_number(0.0f, 0.0f);
    }
}

inline dual_number sigmoid(const dual_number& x) {
    // sigmoid(x) = 1 / (1 + e^(-x))
    float exp_neg_x = std::exp(-x.value());
    float sig_val = 1.0f / (1.0f + exp_neg_x);
    float sig_deriv = sig_val * (1.0f - sig_val);
    return dual_number(sig_val, sig_deriv * x.dual());
}

inline dual_number tanh(const dual_number& x) {
    // tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))
    float tanh_val = std::tanh(x.value());
    float tanh_deriv = 1.0f - tanh_val * tanh_val;
    return dual_number(tanh_val, tanh_deriv * x.dual());
}

// dual_vector class
class dual_vector {
private:
    std::vector<dual_number> data_;

public:
    // Constructors
    dual_vector() {}
    
    explicit dual_vector(size_t size) : data_(size) {}
    
    dual_vector(size_t size, const dual_number& value) : data_(size, value) {}
    
    dual_vector(const std::vector<dual_number>& data) : data_(data) {}
    
    // Element access
    dual_number& operator[](size_t index) {
        return data_[index];
    }
    
    const dual_number& operator[](size_t index) const {
        return data_[index];
    }
    
    // Size
    size_t size() const {
        return data_.size();
    }
    
    // Vector operations
    dual_vector operator+(const dual_vector& rhs) const {
        if (size() != rhs.size()) {
            throw std::invalid_argument("Vector sizes don't match for addition");
        }
        
        dual_vector result(size());
        for (size_t i = 0; i < size(); ++i) {
            result[i] = data_[i] + rhs[i];
        }
        return result;
    }
    
    dual_vector operator-(const dual_vector& rhs) const {
        if (size() != rhs.size()) {
            throw std::invalid_argument("Vector sizes don't match for subtraction");
        }
        
        dual_vector result(size());
        for (size_t i = 0; i < size(); ++i) {
            result[i] = data_[i] - rhs[i];
        }
        return result;
    }
    
    // Element-wise multiplication
    dual_vector operator*(const dual_vector& rhs) const {
        if (size() != rhs.size()) {
            throw std::invalid_argument("Vector sizes don't match for multiplication");
        }
        
        dual_vector result(size());
        for (size_t i = 0; i < size(); ++i) {
            result[i] = data_[i] * rhs[i];
        }
        return result;
    }
    
    // Scalar multiplication
    dual_vector operator*(const dual_number& scalar) const {
        dual_vector result(size());
        for (size_t i = 0; i < size(); ++i) {
            result[i] = data_[i] * scalar;
        }
        return result;
    }
    
    // Vector operations with assignment
    dual_vector& operator+=(const dual_vector& rhs) {
        if (size() != rhs.size()) {
            throw std::invalid_argument("Vector sizes don't match for addition");
        }
        
        for (size_t i = 0; i < size(); ++i) {
            data_[i] += rhs[i];
        }
        return *this;
    }
    
    dual_vector& operator-=(const dual_vector& rhs) {
        if (size() != rhs.size()) {
            throw std::invalid_argument("Vector sizes don't match for subtraction");
        }
        
        for (size_t i = 0; i < size(); ++i) {
            data_[i] -= rhs[i];
        }
        return *this;
    }
    
    // Element-wise application of functions
    friend dual_vector sin(const dual_vector& v) {
        dual_vector result(v.size());
        for (size_t i = 0; i < v.size(); ++i) {
            result[i] = sin(v[i]);
        }
        return result;
    }
    
    friend dual_vector cos(const dual_vector& v) {
        dual_vector result(v.size());
        for (size_t i = 0; i < v.size(); ++i) {
            result[i] = cos(v[i]);
        }
        return result;
    }
    
    friend dual_vector exp(const dual_vector& v) {
        dual_vector result(v.size());
        for (size_t i = 0; i < v.size(); ++i) {
            result[i] = exp(v[i]);
        }
        return result;
    }
    
    friend dual_vector ln(const dual_vector& v) {
        dual_vector result(v.size());
        for (size_t i = 0; i < v.size(); ++i) {
            result[i] = ln(v[i]);
        }
        return result;
    }
    
    friend dual_vector relu(const dual_vector& v) {
        dual_vector result(v.size());
        for (size_t i = 0; i < v.size(); ++i) {
            result[i] = relu(v[i]);
        }
        return result;
    }
    
    friend dual_vector sigmoid(const dual_vector& v) {
        dual_vector result(v.size());
        for (size_t i = 0; i < v.size(); ++i) {
            result[i] = sigmoid(v[i]);
        }
        return result;
    }
    
    friend dual_vector tanh(const dual_vector& v) {
        dual_vector result(v.size());
        for (size_t i = 0; i < v.size(); ++i) {
            result[i] = tanh(v[i]);
        }
        return result;
    }
};

#endif // DUAL_NUMBER_H 