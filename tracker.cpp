#include <pybind11/pybind11.h>
#include <unordered_map>
#include <string>
#include <shared_mutex>

namespace py = pybind11;

class CppTrackerEngine {
private:
    // Memory map replacing the standard slow database/JSON dictionary
    std::unordered_map<std::string, std::string> state_store;
    mutable std::shared_mutex store_mutex;

public:
    void save(const std::string& sender_id, const std::string& state_json) {
        std::unique_lock<std::shared_mutex> lock(store_mutex);
        state_store[sender_id] = state_json;
    }

    std::string retrieve(const std::string& sender_id) const {
        std::shared_lock<std::shared_mutex> lock(store_mutex);
        auto it = state_store.find(sender_id);
        if (it != state_store.end()) {
            return it->second;
        }
        return "";
    }
};

// Bindings to expose this C++ class directly to Python scripts
PYBIND11_MODULE(fast_tracker, m) {
    py::class_<CppTrackerEngine>(m, "CppTrackerEngine")
        .def(py::init<>())
        .def("save", &CppTrackerEngine::save)
        .def("retrieve", &CppTrackerEngine::retrieve);
}