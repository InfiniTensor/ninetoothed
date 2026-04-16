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
        load();

        auto iter = cache.find(config);

        if (iter != cache.end()) {
            return iter->second;
        }

        return default_meta;
    }

private:
    void load() {
        cache.clear();

        std::ifstream file{path};
        std::string config_line;
        std::string meta_line;

        if (std::getline(file, config_line) && std::getline(file, meta_line)) {
            default_meta = parse(meta_line);
        }

        while (std::getline(file, config_line) && std::getline(file, meta_line)) {
            cache[parse(config_line)] = parse(meta_line);
        }
    }

    static std::vector<int> parse(const std::string& line) {
        std::vector<int> values;
        std::stringstream ss{line};
        std::string token;

        while (std::getline(ss, token, ',')) {
            auto trimmed = trim(token);

            if (!trimmed.empty()) {
                values.push_back(std::stoi(trimmed));
            }
        }

        return values;
    }

    static std::string trim(const std::string& s) {
        auto start = s.find_first_not_of(" \t\r\n");

        if (start == std::string::npos) {
            return "";
        }

        auto end = s.find_last_not_of(" \t\r\n");

        return s.substr(start, end - start + 1);
    }

    std::string path;

    std::map<std::vector<int>, std::vector<int>> cache;

    std::vector<int> default_meta;
};

}
