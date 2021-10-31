#pragma once

#include <vector>

#include "spdlog/sinks/basic_file_sink.h"
#include "spdlog/sinks/stdout_color_sinks.h"
#include "spdlog/spdlog.h"


namespace rxmesh {

class Log
{
   public:
    static void init(spdlog::level::level_enum level = spdlog::level::trace)
    {
        std::vector<spdlog::sink_ptr> sinks;
        sinks.emplace_back(
            std::make_shared<spdlog::sinks::stdout_color_sink_mt>());
        sinks.emplace_back(std::make_shared<spdlog::sinks::basic_file_sink_mt>(
            "RXMesh.log", true));

        sinks[0]->set_pattern("%^[%T] %n: %v%$");
        sinks[1]->set_pattern("[%T] [%l] %n: %v");

        m_logger = std::make_shared<spdlog::logger>(
            "RXMesh", begin(sinks), end(sinks));
        spdlog::register_logger(m_logger);
        m_logger->set_level(level);
        m_logger->flush_on(level);
    }

    inline static std::shared_ptr<spdlog::logger>& get_logger()
    {
        return m_logger;
    }


   private:
    inline static std::shared_ptr<spdlog::logger> m_logger;
};
}  // namespace rxmesh

#define RXMESH_TRACE(...) ::rxmesh::Log::get_logger()->trace(__VA_ARGS__)
#define RXMESH_INFO(...) ::rxmesh::Log::get_logger()->info(__VA_ARGS__)
#define RXMESH_WARN(...)                                                      \
    ::rxmesh::Log::get_logger()->warn("Line {} File {}", __LINE__, __FILE__); \
    ::rxmesh::Log::get_logger()->warn(__VA_ARGS__)
#define RXMESH_ERROR(...)                                                      \
    ::rxmesh::Log::get_logger()->error("Line {} File {}", __LINE__, __FILE__); \
    ::rxmesh::Log::get_logger()->error(__VA_ARGS__)
#define RXMESH_CRITICAL(...)                    \
    ::rxmesh::Log::get_logger()->critical(      \
        "Line {} File {}", __LINE__, __FILE__); \
    ::rxmesh::Log::get_logger()->critical(__VA_ARGS__)
