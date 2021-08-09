#pragma once

#include <vector>

#include "spdlog/sinks/basic_file_sink.h"
#include "spdlog/sinks/stdout_color_sinks.h"
#include "spdlog/spdlog.h"


namespace RXMESH {

class Log
{
   public:
    static void init()
    {
        std::vector<spdlog::sink_ptr> sinks;
        sinks.emplace_back(
            std::make_shared<spdlog::sinks::stdout_color_sink_mt>());
        sinks.emplace_back(std::make_shared<spdlog::sinks::basic_file_sink_mt>(
            "RXMesh.log", true));

        sinks[0]->set_pattern("%^[%T] %n: %v%$");
        sinks[1]->set_pattern("[%T] [%l] %n: %v");

        m_logger = std::make_shared<spdlog::logger>("RXMesh", begin(sinks),
                                                    end(sinks));
        spdlog::register_logger(m_logger);
        m_logger->set_level(spdlog::level::trace);
        m_logger->flush_on(spdlog::level::trace);
    }

    inline static std::shared_ptr<spdlog::logger>& get_logger()
    {
        return m_logger;
    }


   private:
    inline static std::shared_ptr<spdlog::logger> m_logger;
};
}  // namespace RXMESH

#define RXMESH_TRACE(...) ::RXMESH::Log::get_logger()->trace(__VA_ARGS__)
#define RXMESH_INFO(...) ::RXMESH::Log::get_logger()->info(__VA_ARGS__)
#define RXMESH_WARN(...)                                                      \
    ::RXMESH::Log::get_logger()->warn("Line {} File {}", __LINE__, __FILE__); \
    ::RXMESH::Log::get_logger()->warn(__VA_ARGS__)
#define RXMESH_ERROR(...)                                                      \
    ::RXMESH::Log::get_logger()->error("Line {} File {}", __LINE__, __FILE__); \
    ::RXMESH::Log::get_logger()->error(__VA_ARGS__)
#define RXMESH_CRITICAL(...)                                           \
    ::RXMESH::Log::get_logger()->critical("Line {} File {}", __LINE__, \
                                          __FILE__);                   \
    ::RXMESH::Log::get_logger()->critical(__VA_ARGS__)
