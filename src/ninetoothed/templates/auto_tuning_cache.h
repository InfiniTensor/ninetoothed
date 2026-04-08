#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

namespace ninetoothed {

class AutoTuningCache {
public:
    AutoTuningCache(const std::string& path) : path{path} {}

    std::vector<int> lookup(const std::vector<int>& config) {
        if (!loaded) {
            load();
        }

        auto iter = cache.find(config);

        if (iter != cache.end()) {
            return iter->second;
        }

        return {};
    }

private:
    void load() {
        std::ifstream file{path};
        std::string config_line;
        std::string meta_line;

        while (std::getline(file, config_line) && std::getline(file, meta_line)) {
            cache[parse(config_line)] = parse(meta_line);
        }

        loaded = true;
    }

    static std::vector<int> parse(const std::string& line) {
        std::vector<int> values;
        std::stringstream ss{line};
        std::string token;

        while (std::getline(ss, token, ',')) {
            values.push_back(std::stoi(token));
        }

        return values;
    }

    std::string path;

    bool loaded{false};

    std::map<std::vector<int>, std::vector<int>> cache;
};

}
