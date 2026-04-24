#include <mutex>
#include <shared_mutex>
#include <unordered_map>

namespace ninetoothed {

template<typename Key, typename Value>
class ThreadSafeUnorderedMap {
public:
    Value& operator[](const Key& key) {
        {
            std::shared_lock lock{mutex};

            auto iter = map.find(key);

            if (iter != map.end()) {
                return iter->second;
            }
        }

        std::unique_lock lock{mutex};

        auto iter = map.find(key);

        if (iter == map.end()) {
            iter = map.emplace(key, Value{}).first;
        }

        return iter->second;
    }

private:
    std::unordered_map<Key, Value> map;

    mutable std::shared_mutex mutex;
};

}
