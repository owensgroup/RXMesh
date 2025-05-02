#pragma once
#include <tuple>

#include <cuComplex.h>

namespace rxmesh {
namespace detail {

/**
 * @brief extracting the input parameter type and return type of a lambda
 * function. Taken from https://stackoverflow.com/a/7943765/1608232.
 * For generic types, directly use the result of the signature of its operator()
 */
template <typename T>
struct FunctionTraits : public FunctionTraits<decltype(&T::operator())>
{
};


template <typename ReturnType, typename... Args>
struct FunctionTraitsBase
{
    static constexpr std::size_t arity = sizeof...(Args);
    using result_type                  = ReturnType;

    /**
     * @brief the i-th argument is equivalent to the i-th tuple element of a
     * tuple composed of those arguments.
     */
    template <std::size_t i>
    struct arg
    {
        using type_rc =
            typename std::tuple_element<i, std::tuple<Args...>>::type;
        using type_c = std::conditional_t<std::is_reference_v<type_rc>,
                                          std::remove_reference_t<type_rc>,
                                          type_rc>;
        using type   = std::conditional_t<std::is_const_v<type_c>,
                                          std::remove_const_t<type_c>,
                                          type_c>;
    };
};


// const-qualified operator()
template <typename ClassType, typename ReturnType, typename... Args>
struct FunctionTraits<ReturnType (ClassType::*)(Args...) const>
    : FunctionTraitsBase<ReturnType, Args...>
{
};

// non-const operator() (for mutable lambdas)
template <typename ClassType, typename ReturnType, typename... Args>
struct FunctionTraits<ReturnType (ClassType::*)(Args...)>
    : FunctionTraitsBase<ReturnType, Args...>
{
};
}  // namespace detail


/**
 * @brief Extracting base type from a type. Used primarily to extract the float
 * and double base type of cuComplex and cuDoubleComplex types
 */
template <typename T>
struct BaseType
{
    using type = T;
};

template <>
struct BaseType<cuComplex>
{
    using type = float;
};

template <>
struct BaseType<cuDoubleComplex>
{
    using type = double;
};


template <typename T>
using BaseTypeT = typename BaseType<T>::type;


namespace detail {

/**
 * @brief Extracting base type from a type. Used primarily to extract the
 * PassiveType from a Scalar
 * Primary template: Default case where T::Passive does not exist
 */

template <typename T, typename = void>
struct is_scalar : std::false_type
{
    using type = T;
};

/**
 * @brief Specialization: When T::PassiveType exists
 */
template <typename T>
struct is_scalar<T, std::void_t<typename T::PassiveType>> : std::true_type
{
    using type = typename T::PassiveType;
};
}  // namespace detail

/**
 * @brief check if a type is a Scalar
 */
template <typename T>
inline constexpr bool is_scalar_v = detail::is_scalar<T>::value;


/**
 * @brief Extracting base type from a type. Used primarily to extract the
 * PassiveType from a Scalar, i.e., if T is a Scalar type, then the returned
 * type is Scalar::PassiveType. If T is a type that does not have PassiveType
 * (e.g., float, double), then the returned type is T itself.
 */
template <typename T>
using PassiveType = typename detail::is_scalar<T>::type;


}  // namespace rxmesh