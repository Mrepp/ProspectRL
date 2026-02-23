-- ProspectRL Turtle Client
-- ComputerCraft / CC:Tweaked Lua program that interfaces with the
-- inference server to get RL-driven mining action decisions.
--
-- Requirements:
--   - CC:Tweaked turtle with a pickaxe (any tier)
--   - HTTP API enabled in CC:Tweaked config
--   - Inference server running and reachable
--
-- Testing (manual):
--   1. Start the inference server:
--        python -m prospect_rl.deployment.inference_server
--   2. Place a CC:Tweaked turtle in a Minecraft world
--   3. Copy this file to the turtle (e.g. via pastebin or local file)
--   4. Run: turtle_client
--   5. Observe the turtle mining with RL-driven decisions
--
--   To test with a mock server, use any HTTP server that responds with
--   valid JSON on POST /act, e.g.:
--     {"action": 0, "action_name": "forward", "confidence": 0.9}
--
-- Configuration can be changed by editing the variables below.

-- =========================================================================
-- Configuration
-- =========================================================================

local SERVER_URL = "http://localhost:8080"
local FUEL_THRESHOLD = 100       -- auto-refuel below this level
local INVENTORY_FULL_THRESHOLD = 14  -- drop junk when this many slots used
local REQUEST_TIMEOUT = 5        -- seconds
local RETRY_DELAY = 2            -- seconds between retries
local MAX_RETRIES = 3            -- max HTTP retries before fallback

-- Target ore (set via command line arg or default)
local TARGET_ORE = arg and arg[1] or "diamond"

-- =========================================================================
-- State tracking (dead-reckoning)
-- =========================================================================

local pos = { x = 0, y = 0, z = 0 }  -- relative to start
local facing = 0  -- 0=north(+z), 1=east(+x), 2=south(-z), 3=west(-x)

-- Facing direction vectors
local FACING_DX = { [0] = 0,  [1] = 1,  [2] = 0,  [3] = -1 }
local FACING_DZ = { [0] = 1,  [1] = 0,  [2] = -1, [3] = 0 }

-- =========================================================================
-- Ore name mappings (block names from CC:Tweaked inspect)
-- =========================================================================

local ORE_BLOCKS = {
    ["minecraft:coal_ore"]     = "coal_ore",
    ["minecraft:deepslate_coal_ore"] = "coal_ore",
    ["minecraft:iron_ore"]     = "iron_ore",
    ["minecraft:deepslate_iron_ore"] = "iron_ore",
    ["minecraft:gold_ore"]     = "gold_ore",
    ["minecraft:deepslate_gold_ore"] = "gold_ore",
    ["minecraft:diamond_ore"]  = "diamond_ore",
    ["minecraft:deepslate_diamond_ore"] = "diamond_ore",
    ["minecraft:redstone_ore"] = "redstone_ore",
    ["minecraft:deepslate_redstone_ore"] = "redstone_ore",
    ["minecraft:emerald_ore"]  = "emerald_ore",
    ["minecraft:deepslate_emerald_ore"] = "emerald_ore",
    ["minecraft:lapis_ore"]    = "lapis_ore",
    ["minecraft:deepslate_lapis_ore"] = "lapis_ore",
}

local NON_ORE_JUNK = {
    ["minecraft:cobblestone"] = true,
    ["minecraft:dirt"]        = true,
    ["minecraft:gravel"]      = true,
    ["minecraft:sand"]        = true,
    ["minecraft:netherrack"]  = true,
    ["minecraft:cobbled_deepslate"] = true,
    ["minecraft:tuff"]        = true,
    ["minecraft:andesite"]    = true,
    ["minecraft:diorite"]     = true,
    ["minecraft:granite"]     = true,
}

-- =========================================================================
-- Utility functions
-- =========================================================================

local function log(msg)
    print("[RL-Miner] " .. msg)
end

local function json_encode(tbl)
    return textutils.serialiseJSON(tbl)
end

local function json_decode(str)
    return textutils.unserialiseJSON(str)
end

-- =========================================================================
-- Task 9.2: State collection functions
-- =========================================================================

--- Translate a CC:Tweaked block name to our internal block name.
local function translate_block(cc_name)
    if cc_name == nil or cc_name == "" then
        return "air"
    end
    if ORE_BLOCKS[cc_name] then
        return ORE_BLOCKS[cc_name]
    end
    if cc_name == "minecraft:air" or cc_name == "minecraft:cave_air" then
        return "air"
    end
    if cc_name == "minecraft:bedrock" then
        return "bedrock"
    end
    if cc_name == "minecraft:dirt" or cc_name == "minecraft:grass_block" then
        return "dirt"
    end
    -- Default to stone for any unknown solid block
    return "stone"
end

--- Inspect blocks around the turtle.
-- CC:Tweaked only allows inspecting front, up, and down.
-- All other positions in the 5x5x5 cube default to "stone" (safe assumption
-- underground -- the model was trained this way).
local function collect_nearby_blocks()
    local size = 5
    local total = size * size * size  -- 125
    local blocks = {}
    for i = 1, total do
        blocks[i] = "stone"
    end

    -- Centre of the 5x5x5 cube (1-indexed): (3, 3, 3)
    -- Lua arrays are 1-indexed; the flat index for (x,y,z) is:
    --   idx = (x-1)*25 + (y-1)*5 + z
    -- Centre = (3-1)*25 + (3-1)*5 + 3 = 63

    -- The turtle's own position is air
    blocks[63] = "air"

    -- Front block (facing direction, offset = facing_vec at centre)
    local ok, data = turtle.inspect()
    if ok then
        local dx, dy, dz = FACING_DX[facing], 0, FACING_DZ[facing]
        local fx = 3 + dx
        local fy = 3 + dy
        local fz = 3 + dz
        local fi = (fx - 1) * 25 + (fy - 1) * 5 + fz
        blocks[fi] = translate_block(data.name)
    else
        -- No block in front => air
        local dx, dy, dz = FACING_DX[facing], 0, FACING_DZ[facing]
        local fx = 3 + dx
        local fy = 3 + dy
        local fz = 3 + dz
        local fi = (fx - 1) * 25 + (fy - 1) * 5 + fz
        blocks[fi] = "air"
    end

    -- Block above
    local ok_up, data_up = turtle.inspectUp()
    if ok_up then
        -- Up is (3, 4, 3) => idx = 2*25 + 3*5 + 3 = 68
        blocks[68] = translate_block(data_up.name)
    else
        blocks[68] = "air"
    end

    -- Block below
    local ok_down, data_down = turtle.inspectDown()
    if ok_down then
        -- Down is (3, 2, 3) => idx = 2*25 + 1*5 + 3 = 58
        blocks[58] = translate_block(data_down.name)
    else
        blocks[58] = "air"
    end

    return blocks
end

--- Scan inventory and return ore counts.
local function collect_inventory()
    local inv = {}
    for slot = 1, 16 do
        local detail = turtle.getItemDetail(slot)
        if detail then
            local ore_name = ORE_BLOCKS[detail.name]
            if ore_name then
                -- Strip _ore suffix for the API (e.g. "coal_ore" -> "coal")
                local short = ore_name:gsub("_ore$", "")
                inv[short] = (inv[short] or 0) + detail.count
            end
        end
    end
    return inv
end

--- Count how many inventory slots are used.
local function count_used_slots()
    local used = 0
    for slot = 1, 16 do
        if turtle.getItemCount(slot) > 0 then
            used = used + 1
        end
    end
    return used
end

--- Try to get absolute position via GPS, fall back to dead-reckoning.
local function get_position()
    if gps then
        local x, y, z = gps.locate(2)
        if x then
            return { x, y, z }
        end
    end
    -- Dead-reckoning
    return { pos.x, pos.y, pos.z }
end

--- Collect the full state payload for the /act endpoint.
local function collect_state()
    return {
        position = get_position(),
        facing = facing,
        fuel = turtle.getFuelLevel(),
        inventory = collect_inventory(),
        nearby_blocks = collect_nearby_blocks(),
    }
end

-- =========================================================================
-- HTTP communication
-- =========================================================================

--- Send an HTTP POST request with JSON body. Returns decoded response or nil.
local function http_post(path, body)
    local url = SERVER_URL .. path
    local json_body = json_encode(body)

    for attempt = 1, MAX_RETRIES do
        local response = http.post(url, json_body, {
            ["Content-Type"] = "application/json",
        })
        if response then
            local text = response.readAll()
            response.close()
            local data = json_decode(text)
            if data then
                return data
            end
        end
        if attempt < MAX_RETRIES then
            log("HTTP request failed (attempt " .. attempt .. "), retrying...")
            os.sleep(RETRY_DELAY)
        end
    end

    return nil
end

--- Request the preference vector for our target ore.
local function get_current_preference()
    local resp = http_post("/preference", { target = TARGET_ORE })
    if resp and resp.preference then
        return resp.preference
    end
    -- Fallback: diamond one-hot
    log("Could not fetch preference, using default diamond target")
    return { 0, 0, 0, 1, 0, 0, 0 }
end

--- Request an action from the inference server.
local function request_action(state, preference)
    local payload = {
        position = state.position,
        facing = state.facing,
        fuel = state.fuel,
        inventory = state.inventory,
        nearby_blocks = state.nearby_blocks,
        preference = preference,
    }
    return http_post("/act", payload)
end

-- =========================================================================
-- Action execution
-- =========================================================================

--- Execute the action returned by the server. Updates dead-reckoning state.
local function execute_action(action_id)
    if action_id == 0 then
        -- forward
        if turtle.forward() then
            pos.x = pos.x + FACING_DX[facing]
            pos.z = pos.z + FACING_DZ[facing]
        end
    elseif action_id == 1 then
        -- back
        if turtle.back() then
            pos.x = pos.x - FACING_DX[facing]
            pos.z = pos.z - FACING_DZ[facing]
        end
    elseif action_id == 2 then
        -- up
        if turtle.up() then
            pos.y = pos.y + 1
        end
    elseif action_id == 3 then
        -- down
        if turtle.down() then
            pos.y = pos.y - 1
        end
    elseif action_id == 4 then
        -- turn left
        turtle.turnLeft()
        facing = (facing - 1) % 4
    elseif action_id == 5 then
        -- turn right
        turtle.turnRight()
        facing = (facing + 1) % 4
    elseif action_id == 6 then
        -- dig (front)
        turtle.dig()
    elseif action_id == 7 then
        -- dig up
        turtle.digUp()
    elseif action_id == 8 then
        -- dig down
        turtle.digDown()
    else
        log("Unknown action: " .. tostring(action_id))
    end
end

-- =========================================================================
-- Task 9.3: Fallback behaviors
-- =========================================================================

--- Auto-refuel from inventory when fuel is low.
local function handle_fuel()
    local fuel_level = turtle.getFuelLevel()
    if fuel_level == "unlimited" then
        return
    end
    if fuel_level >= FUEL_THRESHOLD then
        return
    end

    log("Low fuel (" .. fuel_level .. "), attempting to refuel...")
    local current_slot = turtle.getSelectedSlot()

    for slot = 1, 16 do
        local detail = turtle.getItemDetail(slot)
        if detail then
            -- Refuel from coal/charcoal/lava buckets etc.
            turtle.select(slot)
            if turtle.refuel(1) then
                log("Refueled from slot " .. slot .. ", fuel now: " .. turtle.getFuelLevel())
                if turtle.getFuelLevel() >= FUEL_THRESHOLD then
                    turtle.select(current_slot)
                    return
                end
            end
        end
    end

    turtle.select(current_slot)

    if turtle.getFuelLevel() < 10 then
        log("CRITICAL: Fuel very low (" .. turtle.getFuelLevel() .. "), cannot continue safely")
    end
end

--- Drop non-ore items when inventory is nearly full.
local function handle_inventory()
    local used = count_used_slots()
    if used < INVENTORY_FULL_THRESHOLD then
        return
    end

    log("Inventory nearly full (" .. used .. "/16 slots), dropping junk...")
    local current_slot = turtle.getSelectedSlot()

    for slot = 1, 16 do
        local detail = turtle.getItemDetail(slot)
        if detail and NON_ORE_JUNK[detail.name] then
            turtle.select(slot)
            turtle.drop()
            log("Dropped " .. detail.name .. " from slot " .. slot)
        end
    end

    turtle.select(current_slot)
end

-- =========================================================================
-- Main loop
-- =========================================================================

local function main()
    log("ProspectRL Client starting...")
    log("Target ore: " .. TARGET_ORE)
    log("Server: " .. SERVER_URL)

    -- Fetch the preference vector for our target ore
    local preference = get_current_preference()
    log("Preference vector received")

    local step = 0
    while true do
        step = step + 1

        -- Collect current state
        local state = collect_state()

        -- Request action from server
        local response = request_action(state, preference)

        if response and response.action then
            execute_action(response.action)
            if step % 50 == 0 then
                log("Step " .. step
                    .. " | Action: " .. (response.action_name or "?")
                    .. " | Fuel: " .. turtle.getFuelLevel()
                    .. " | Conf: " .. string.format("%.2f", response.confidence or 0))
            end
        else
            log("No response from server, waiting...")
            os.sleep(RETRY_DELAY)
        end

        -- Client-side fallback behaviors
        handle_fuel()
        handle_inventory()

        -- Small yield to prevent CC:Tweaked "too long without yielding" error
        os.sleep(0.05)
    end
end

-- Run
main()
