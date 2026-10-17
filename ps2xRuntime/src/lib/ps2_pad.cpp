#include "runtime/ps2_pad.h"
#include "ps2_host_backend.h"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>

#if !defined(PLATFORM_VITA)
// raylib's desktop platform is built on the GLFW it bundles; these are not exposed by raylib.h.
extern "C"
{
    int glfwJoystickPresent(int jid);
    int glfwJoystickIsGamepad(int jid);
    const char *glfwGetJoystickGUID(int jid);
    const char *glfwGetJoystickName(int jid);
}
#endif

namespace
{
    constexpr uint8_t kPadAnalogMarker = 0x73;
    constexpr uint8_t kPadStickCenter = 0x80;
    // Radial stick deadzone; the remaining travel is rescaled to the full range.
    constexpr float kStickDeadzone = 0.15f;
    constexpr int kMaxGamepads = 4;

    constexpr uint16_t PAD_LEFT = 0x0080u;
    constexpr uint16_t PAD_DOWN = 0x0040u;
    constexpr uint16_t PAD_RIGHT = 0x0020u;
    constexpr uint16_t PAD_UP = 0x0010u;
    constexpr uint16_t PAD_START = 0x0008u;
    constexpr uint16_t PAD_R3 = 0x0004u;
    constexpr uint16_t PAD_L3 = 0x0002u;
    constexpr uint16_t PAD_SELECT = 0x0001u;
    constexpr uint16_t PAD_SQUARE = 0x8000u;
    constexpr uint16_t PAD_CROSS = 0x4000u;
    constexpr uint16_t PAD_CIRCLE = 0x2000u;
    constexpr uint16_t PAD_TRIANGLE = 0x1000u;
    constexpr uint16_t PAD_R1 = 0x0800u;
    constexpr uint16_t PAD_L1 = 0x0400u;
    constexpr uint16_t PAD_R2 = 0x0200u;
    constexpr uint16_t PAD_L2 = 0x0100u;

    uint8_t axisToByte(float axis)
    {
        axis = std::clamp(axis, -1.0f, 1.0f);
        return static_cast<uint8_t>(std::lround((axis + 1.0f) * 127.5f));
    }

    void applyDeadzone(float &x, float &y)
    {
        const float magnitude = std::sqrt(x * x + y * y);
        if (magnitude <= kStickDeadzone)
        {
            x = y = 0.0f;
            return;
        }
        const float scale = std::min(1.0f, (magnitude - kStickDeadzone) / (1.0f - kStickDeadzone)) / magnitude;
        x *= scale;
        y *= scale;
    }

#if !defined(PLATFORM_VITA)
    // GLFW's built-in database only lists some firmware versions of the Xbox Series controller over
    // Bluetooth (the version is part of the GUID). Newer firmware keeps the same layout, so any
    // Bluetooth 045e:0b13 controller without a mapping gets this one.
    constexpr const char *kXboxSeriesBtPrefix = "050000005e040000130b0000";
    constexpr const char *kXboxSeriesBtMapping =
        "Xbox Series Controller,a:b0,b:b1,back:b10,dpdown:h0.4,dpleft:h0.8,dpright:h0.2,dpup:h0.1,guide:b12,"
        "leftshoulder:b6,leftstick:b13,lefttrigger:a5,leftx:a0,lefty:a1,rightshoulder:b7,rightstick:b14,"
        "righttrigger:a4,rightx:a2,righty:a3,start:b11,x:b3,y:b4,platform:Linux,";

    // SDL-format mappings from SDL_GAMECONTROLLERCONFIG and the file named by PS2X_GAMECONTROLLERDB
    // (default ./gamecontrollerdb.txt).
    void loadUserMappings()
    {
        if (const char *env = std::getenv("SDL_GAMECONTROLLERCONFIG"); env && *env)
            SetGamepadMappings(env);

        const char *path = std::getenv("PS2X_GAMECONTROLLERDB");
        std::ifstream file(path && *path ? path : "gamecontrollerdb.txt");
        if (!file)
            return;
        std::stringstream contents;
        contents << file.rdbuf();
        if (SetGamepadMappings(contents.str().c_str()))
            std::cout << "[pad] loaded controller mappings from " << (path && *path ? path : "gamecontrollerdb.txt") << std::endl;
    }
#endif
}

void PSPadBackend::updateGamepads()
{
#if defined(PLATFORM_VITA)
    m_activeGamepad.store(IsGamepadAvailable(0) ? 0 : -1, std::memory_order_relaxed);
#else
    if (!m_userMappingsLoaded)
    {
        m_userMappingsLoaded = true;
        loadUserMappings();
    }

    int active = -1;
    for (int jid = 0; jid < kMaxGamepads; ++jid)
    {
        if (!glfwJoystickPresent(jid))
        {
            m_seenGuids[jid].clear();
            continue;
        }

        const char *guidText = glfwGetJoystickGUID(jid);
        const std::string guid = guidText ? guidText : "";
        if (guid != m_seenGuids[jid])
        {
            m_seenGuids[jid] = guid;
            const char *nameText = glfwGetJoystickName(jid);
            const std::string name = nameText ? nameText : "?";
            if (!glfwJoystickIsGamepad(jid) && guid.rfind(kXboxSeriesBtPrefix, 0) == 0)
                SetGamepadMappings((guid + "," + kXboxSeriesBtMapping).c_str());

            if (glfwJoystickIsGamepad(jid))
                std::cout << "[pad] controller " << jid << ": " << name << " (" << guid << ")" << std::endl;
            else
                std::cout << "[pad] controller " << jid << ": " << name << " (" << guid
                          << ") has no gamepad mapping; add one via SDL_GAMECONTROLLERCONFIG or PS2X_GAMECONTROLLERDB"
                          << std::endl;
        }

        if (active < 0 && glfwJoystickIsGamepad(jid))
            active = jid;
    }
    m_activeGamepad.store(active, std::memory_order_relaxed);
#endif
}

bool PSPadBackend::readState(int /*port*/, int /*slot*/, uint8_t *data, size_t size)
{
    if (!data || size < 32)
        return false;

    std::memset(data, 0, 32);
    data[0] = 0x01;
    data[1] = kPadAnalogMarker;

    uint16_t btns = 0xFFFFu;
    auto press = [&btns](uint16_t mask, bool down)
    {
        if (down)
            btns &= ~mask;
    };
    float lx = 0.0f, ly = 0.0f, rx = 0.0f, ry = 0.0f;

    // Controller and keyboard are merged, so either can be used at any time.
    const int gamepad = m_activeGamepad.load(std::memory_order_relaxed);
    if (gamepad >= 0 && IsGamepadAvailable(gamepad))
    {
        press(PAD_UP, IsGamepadButtonDown(gamepad, GAMEPAD_BUTTON_LEFT_FACE_UP));
        press(PAD_DOWN, IsGamepadButtonDown(gamepad, GAMEPAD_BUTTON_LEFT_FACE_DOWN));
        press(PAD_LEFT, IsGamepadButtonDown(gamepad, GAMEPAD_BUTTON_LEFT_FACE_LEFT));
        press(PAD_RIGHT, IsGamepadButtonDown(gamepad, GAMEPAD_BUTTON_LEFT_FACE_RIGHT));
        press(PAD_CROSS, IsGamepadButtonDown(gamepad, GAMEPAD_BUTTON_RIGHT_FACE_DOWN));
        press(PAD_CIRCLE, IsGamepadButtonDown(gamepad, GAMEPAD_BUTTON_RIGHT_FACE_RIGHT));
        press(PAD_SQUARE, IsGamepadButtonDown(gamepad, GAMEPAD_BUTTON_RIGHT_FACE_LEFT));
        press(PAD_TRIANGLE, IsGamepadButtonDown(gamepad, GAMEPAD_BUTTON_RIGHT_FACE_UP));
        press(PAD_L1, IsGamepadButtonDown(gamepad, GAMEPAD_BUTTON_LEFT_TRIGGER_1));
        press(PAD_R1, IsGamepadButtonDown(gamepad, GAMEPAD_BUTTON_RIGHT_TRIGGER_1));
        press(PAD_L2, IsGamepadButtonDown(gamepad, GAMEPAD_BUTTON_LEFT_TRIGGER_2));
        press(PAD_R2, IsGamepadButtonDown(gamepad, GAMEPAD_BUTTON_RIGHT_TRIGGER_2));
        press(PAD_START, IsGamepadButtonDown(gamepad, GAMEPAD_BUTTON_MIDDLE_RIGHT));
        press(PAD_SELECT, IsGamepadButtonDown(gamepad, GAMEPAD_BUTTON_MIDDLE_LEFT));
        press(PAD_L3, IsGamepadButtonDown(gamepad, GAMEPAD_BUTTON_LEFT_THUMB));
        press(PAD_R3, IsGamepadButtonDown(gamepad, GAMEPAD_BUTTON_RIGHT_THUMB));

        lx = GetGamepadAxisMovement(gamepad, GAMEPAD_AXIS_LEFT_X);
        ly = GetGamepadAxisMovement(gamepad, GAMEPAD_AXIS_LEFT_Y);
        rx = GetGamepadAxisMovement(gamepad, GAMEPAD_AXIS_RIGHT_X);
        ry = GetGamepadAxisMovement(gamepad, GAMEPAD_AXIS_RIGHT_Y);
        applyDeadzone(lx, ly);
        applyDeadzone(rx, ry);
    }

    // Keyboard: arrows = d-pad, WASD = left stick, IJKL = right stick.
    press(PAD_UP, IsKeyDown(KEY_UP));
    press(PAD_DOWN, IsKeyDown(KEY_DOWN));
    press(PAD_LEFT, IsKeyDown(KEY_LEFT));
    press(PAD_RIGHT, IsKeyDown(KEY_RIGHT));
    press(PAD_CROSS, IsKeyDown(KEY_X) || IsKeyDown(KEY_SPACE));
    press(PAD_CIRCLE, IsKeyDown(KEY_C));
    press(PAD_SQUARE, IsKeyDown(KEY_Z) || IsKeyDown(KEY_KP_0));
    press(PAD_TRIANGLE, IsKeyDown(KEY_V) || IsKeyDown(KEY_KP_1));
    press(PAD_L1, IsKeyDown(KEY_Q));
    press(PAD_R1, IsKeyDown(KEY_E));
    press(PAD_L2, IsKeyDown(KEY_LEFT_SHIFT));
    press(PAD_R2, IsKeyDown(KEY_RIGHT_SHIFT));
    press(PAD_START, IsKeyDown(KEY_ENTER));
    press(PAD_SELECT, IsKeyDown(KEY_TAB));

    auto keyStick = [](float &x, float &y, KeyboardKey left, KeyboardKey right, KeyboardKey up, KeyboardKey down)
    {
        const float kx = (IsKeyDown(right) ? 1.0f : 0.0f) - (IsKeyDown(left) ? 1.0f : 0.0f);
        const float ky = (IsKeyDown(down) ? 1.0f : 0.0f) - (IsKeyDown(up) ? 1.0f : 0.0f);
        if (kx == 0.0f && ky == 0.0f)
            return;
        const float length = std::sqrt(kx * kx + ky * ky);
        x = kx / length;
        y = ky / length;
    };
    keyStick(lx, ly, KEY_A, KEY_D, KEY_W, KEY_S);
    keyStick(rx, ry, KEY_J, KEY_L, KEY_I, KEY_K);

    data[2] = static_cast<uint8_t>(btns & 0xFF);
    data[3] = static_cast<uint8_t>(btns >> 8);
    data[4] = axisToByte(rx);
    data[5] = axisToByte(ry);
    data[6] = axisToByte(lx);
    data[7] = axisToByte(ly);
    return true;
}
