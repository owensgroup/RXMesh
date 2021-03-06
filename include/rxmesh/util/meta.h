#pragma once
#include <tuple>
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

/**
 * @brief specialization for pointers to member function
 */
template <typename ClassType, typename ReturnType, typename... Args>
struct FunctionTraits<ReturnType (ClassType::*)(Args...) const>
{
    /**
     * @brief arity is the number of arguments.
     */
    enum
    {
        arity = sizeof...(Args)
    };

    typedef ReturnType result_type;

    /**
     * @brief the i-th argument is equivalent to the i-th tuple element of a
     * tuple composed of those arguments.
     */
    template <size_t i>
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

}  // namespace detail
}  // namespace rxmesh