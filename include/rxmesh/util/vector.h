#pragma once
#include <assert.h>
#include <stdint.h>

namespace RXMESH {

template <uint32_t N, typename T>
struct Vector
{
    static_assert(N > 0);

    // constructors
    __host__ __device__ __forceinline__ Vector()
    {
    }
    __host__ __device__ __forceinline__ Vector(T value)
    {
        for (uint32_t i = 0; i < N; ++i) {
            m_v[i] = value;
        }
    }

    __host__ __device__ __forceinline__ Vector(const T* source)
    {
        for (uint32_t i = 0; i < N; ++i) {
            m_v[i] = source[i];
        }
    }

    __host__ __device__ __forceinline__ Vector(const Vector<N, T>& source)
    {
        for (uint32_t i = 0; i < N; ++i) {
            m_v[i] = source[i];
        }
    }

    __host__ __device__ __forceinline__ Vector(T a0, T a1)
    {
        assert(N >= 2);
        m_v[0] = a0;
        m_v[1] = a1;
    }

    __host__ __device__ __forceinline__ Vector(T a0, T a1, T a2)
    {
        assert(N >= 3);
        m_v[0] = a0;
        m_v[1] = a1;
        m_v[2] = a2;
    }

    __host__ __device__ __forceinline__ Vector(T a0, T a1, T a2, T a3)
    {
        assert(N >= 4);
        m_v[0] = a0;
        m_v[1] = a1;
        m_v[2] = a2;
        m_v[3] = a3;
    }

    __host__ __device__ __forceinline__ Vector(T a0, T a1, T a2, T a3, T a4)
    {
        assert(N >= 5);
        m_v[0] = a0;
        m_v[1] = a1;
        m_v[2] = a2;
        m_v[3] = a3;
        m_v[4] = a4;
    }

    __host__ __device__ __forceinline__
    Vector(T a0, T a1, T a2, T a3, T a4, T a5)
    {
        assert(N >= 6);
        m_v[0] = a0;
        m_v[1] = a1;
        m_v[2] = a2;
        m_v[3] = a3;
        m_v[4] = a4;
        m_v[5] = a5;
    }

    // static functions
    __host__ __device__ __forceinline__ static Vector<N, T> zero()
    {
        Vector<N, T> ret;
        for (uint32_t i = 0; i < N; ++i) {
            ret.m_v[i] = 0;
        }
        return ret;
    }

    __host__ __device__ __forceinline__ static Vector<N, T> constant(T c)
    {
        Vector<N, T> ret;
        for (uint32_t i = 0; i < N; ++i) {
            ret.m_v[i] = c;
        }
        return ret;
    }

    // indexing
    __host__ __device__ __forceinline__ T& operator[](int index)
    {
        assert(index >= 0 && index < N);
        return m_v[index];
    }

    __host__ __device__ __forceinline__ T operator[](int index) const
    {
        assert(index >= 0 && index < N);
        return m_v[index];
    }

    // unary operators
    __host__ __device__ __forceinline__ const Vector& operator+() const
    {
        return *this;
    }

    __host__ __device__ __forceinline__ Vector operator-() const
    {
        Vector<N, T> ret;
        for (uint32_t i = 0; i < N; ++i) {
            ret.m_v[i] = -m_v[i];
        }
        return ret;
    }


    // binary operators
    // plus
    __host__ __device__ __forceinline__ Vector& operator+=(
        const Vector<N, T>& v)
    {
        for (uint32_t i = 0; i < N; ++i) {
            m_v[i] += v.m_v[i];
        }
        return *this;
    }
    __host__ __device__ __forceinline__ Vector
    operator+(const Vector<N, T>& v) const
    {
        Vector<N, T> ret(*this);
        ret += v;
        return ret;
    }
    template <typename R>
    __host__ __device__ __forceinline__ Vector& operator+=(const R c)
    {
        for (uint32_t i = 0; i < N; ++i) {
            m_v[i] += c;
        }
        return *this;
    }
    template <typename R>
    __host__ __device__ __forceinline__ Vector operator+(const R c) const
    {
        Vector<N, T> ret(*this);
        ret += c;
        return ret;
    }

    // minus
    __host__ __device__ __forceinline__ Vector& operator-=(
        const Vector<N, T>& v)
    {
        for (uint32_t i = 0; i < N; ++i) {
            m_v[i] -= v.m_v[i];
        }
        return *this;
    }
    __host__ __device__ __forceinline__ Vector
    operator-(const Vector<N, T>& v) const
    {
        Vector<N, T> ret(*this);
        ret -= v;
        return ret;
    }
    template <typename R>
    __host__ __device__ __forceinline__ Vector& operator-=(const R c)
    {
        for (uint32_t i = 0; i < N; ++i) {
            m_v[i] -= c;
        }
        return *this;
    }
    template <typename R>
    __host__ __device__ __forceinline__ Vector operator-(const R c) const
    {
        Vector<N, T> ret(*this);
        ret -= c;
        return ret;
    }

    // multiply
    __host__ __device__ __forceinline__ Vector& operator*=(const Vector& v)
    {
        for (uint32_t i = 0; i < N; ++i) {
            m_v[i] *= v.m_v[i];
        }
        return *this;
    }
    __host__ __device__ __forceinline__ Vector operator*(const Vector& v) const
    {
        Vector<N, T> ret(*this);
        ret *= v;
        return ret;
    }
    template <typename R>
    __host__ __device__ __forceinline__ Vector& operator*=(const R c)
    {
        for (uint32_t i = 0; i < N; ++i) {
            m_v[i] *= c;
        }
        return *this;
    }
    template <typename R>
    __host__ __device__ __forceinline__ Vector operator*(const R c) const
    {
        Vector<N, T> ret(*this);
        ret *= c;
        return ret;
    }

    // division
    __host__ __device__ __forceinline__ Vector& operator/=(const Vector& v)
    {
        for (uint32_t i = 0; i < N; ++i) {
            m_v[i] /= v.m_v[i];
        }
        return *this;
    }
    __host__ __device__ __forceinline__ Vector operator/(const Vector& v)
    {
        Vector<N, T> ret(*this);
        ret /= v;
        return ret;
    }
    template <typename R>
    __host__ __device__ __forceinline__ Vector& operator/=(const R c)
    {
        for (uint32_t i = 0; i < N; ++i) {
            m_v[i] /= c;
        }
        return *this;
    }
    template <typename R>
    __host__ __device__ __forceinline__ Vector operator/(const R c)
    {
        Vector<N, T> ret(*this);
        ret /= c;
        return ret;
    }

    // equality
    __host__ __device__ __forceinline__ bool operator==(const Vector& v) const
    {
        for (uint32_t i = 0; i < N; ++i) {
            if (m_v[i] != v.m_v[i]) {
                return false;
            }
        }
        return true;
    }

    __host__ __device__ __forceinline__ bool operator!=(const Vector& v) const
    {
        return !(*this == v);
    }


    // operations
    __host__ __device__ __forceinline__ T norm() const
    {
        T len = 0;
        for (uint32_t i = 0; i < N; ++i) {
            len += m_v[i] * m_v[i];
        }
        return sqrt(len);
    }
    __host__ __device__ __forceinline__ T norm2() const
    {
        T len = 0;
        for (uint32_t i = 0; i < N; ++i) {
            len += m_v[i] * m_v[i];
        }
        return len;
    }
    __host__ __device__ __forceinline__ void normalize()
    {
        T r = norm();
        if(r == T(0.0)){
            for (uint32_t i = 0; i < N; ++i) {
                m_v[i] = 0;
            }
        }else{
            r = 1. / r;
            (*this) *= r;
        }
    }

    __host__ __device__ __forceinline__ T sum()
    {
        T s = 0;
        for (uint32_t i = 0; i < N; ++i) {
            s += m_v[i];
        }
        return s;
    }

    __host__ __device__ __forceinline__ void clamp(T low, T high)
    {
        for (uint32_t i = 0; i < N; ++i) {
            m_v[i] = (m_v[i] <= low) ? low : ((m_v[i] >= high) ? high : m_v[i]);
        }
    }

    __host__ __device__ __forceinline__ T max()
    {
        T m = m_v[0];
        for (uint32_t i = 1; i < N; ++i) {
            m = (m_v[i] > m) ? m_v[i] : m;
        }
        return m;
    }

    __host__ __device__ __forceinline__ T min()
    {
        T m = m_v[0];
        for (uint32_t i = 1; i < N; ++i) {
            m = (m_v[i] < m) ? m_v[i] : m;
        }
        return m;
    }

   private:
    T m_v[N];
};

// operations on vectors
template <uint32_t N, typename T>
__host__ __device__ __forceinline__ T norm(const Vector<N, T>& v)
{
    return v.norm();
}

template <uint32_t N, typename T>
__host__ __device__ __forceinline__ T norm2(const Vector<N, T>& v)
{
    return v.norm2();
}

template <uint32_t N, typename T>
__host__ __device__ __forceinline__ void normalize(Vector<N, T>& v)
{
    v.normalize();
}

template <typename T>
__host__ __device__ __forceinline__ Vector<3, T> cross(const Vector<3, T>& u,
                                                       const Vector<3, T>& v)
{
    T x = u[1] * v[2] - u[2] * v[1];
    T y = u[2] * v[0] - u[0] * v[2];
    T z = u[0] * v[1] - u[1] * v[0];
    return Vector<3, T>{x, y, z};
}


template <typename T>
__host__ __device__ __forceinline__ Vector<2, T> cross(const Vector<2, T>& u,
                                                       const Vector<2, T>& v)
{
    return u[0] * v[1] - u[1] * v[0];
}

template <typename T>
__host__ __device__ __forceinline__ T dot(const Vector<3, T>& u,
                                          const Vector<3, T>& v)
{
    return u[0] * v[0] + u[1] * v[1] + u[2] * v[2];
}

template <typename T>
__host__ __device__ __forceinline__ T dot(const Vector<2, T>& u,
                                          const Vector<2, T>& v)
{
    return u[0] * v[0] + u[1] * v[1];
}

template <uint32_t N, typename T>
__host__ __device__ __forceinline__ T dist2(const Vector<N, T>& u,
                                            const Vector<N, T>& v)
{
    T d = 0;
    for (uint32_t i = 0; i < N; ++i) {
        d += (u[i] - v[i]) * (u[i] - v[i]);
    }
    return d;
}

template <uint32_t N, typename T>
__host__ __device__ __forceinline__ T dist(const Vector<N, T>& u,
                                           const Vector<N, T>& v)
{
    return sqrt(dist2(u, v));
}


template <uint32_t N, typename T>
inline std::string to_string(const Vector<N, T>& v)
{
    std::stringstream ss;
    ss << "[";
    for (uint32_t i = 0; i < N; i++) {
        ss << v[i];
        if (i != N - 1) {
            ss << ", ";
        }
    }
    ss << "]";
    return ss.str();
}

template <uint32_t N, typename T>
inline std::ostream& operator<<(std::ostream& output, const Vector<N, T>& v)
{
    output << to_string(v);
    return output;
}

template <uint32_t N, typename T>
inline std::istream& operator>>(std::istream& input, const Vector<N, T>& v)
{
    for (uint32_t i = 0; i < N; i++) {
        input >> v[i];
    }
    return input;
}

// Alias
using Vector2d = Vector<2, double>;
using Vector2f = Vector<2, float>;
using Vector2i = Vector<2, int32_t>;
using Vector2ui = Vector<2, uint32_t>;
using Vector2s = Vector<2, int16_t>;
using Vector2us = Vector<2, uint16_t>;
using Vector2c = Vector<2, int8_t>;
using Vector2uc = Vector<2, uint8_t>;

using Vector3d = Vector<3, double>;
using Vector3f = Vector<3, float>;
using Vector3i = Vector<3, int32_t>;
using Vector3ui = Vector<3, uint32_t>;
using Vector3s = Vector<3, int16_t>;
using Vector3us = Vector<3, uint16_t>;
using Vector3c = Vector<3, int8_t>;
using Vector3uc = Vector<3, uint8_t>;

using Vector4d = Vector<4, double>;
using Vector4f = Vector<4, float>;
using Vector4i = Vector<4, int32_t>;
using Vector4ui = Vector<4, uint32_t>;
using Vector4s = Vector<4, int16_t>;
using Vector4us = Vector<4, uint16_t>;
using Vector4c = Vector<4, int8_t>;
using Vector4uc = Vector<4, uint8_t>;

using Vector6d = Vector<6, double>;
using Vector6f = Vector<6, float>;
using Vector6i = Vector<6, int32_t>;
using Vector6ui = Vector<6, uint32_t>;
using Vector6s = Vector<6, int16_t>;
using Vector6us = Vector<6, uint16_t>;
using Vector6c = Vector<6, int8_t>;
using Vector6uc = Vector<6, uint8_t>;
}  // namespace RXMESH

// Hash
namespace std {

template <uint32_t N, typename T>
struct hash<RXMESH::Vector<N, T>>
{
    std::size_t operator()(const RXMESH::Vector<N, T>& v) const
    {
        std::size_t h = 0;
        for (int i = 0; i < N; i++) {
            h = std::hash<T>()(v[i]) ^ (h << 1);
        }
        return h;
    }
};

}  // namespace std