#ifndef PS2_PAD_H
#define PS2_PAD_H

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <string>

class PSPadBackend
{
public:
    PSPadBackend() = default;
    ~PSPadBackend() = default;

    bool readState(int port, int slot, uint8_t *data, size_t size);

    // Main thread only (GLFW joystick calls). Registers mappings for controllers GLFW does not know
    // and picks the controller readState uses; call every frame so hot-plugged controllers work.
    void updateGamepads();

private:
    std::atomic<int> m_activeGamepad{-1};
    bool m_userMappingsLoaded = false;
    std::string m_seenGuids[4];
};

#endif
