using AutomotiveDrivingModels
using NNMPC

const EGO_ID = 1

mutable struct HRIParams
    road::Float64 # length of roadway
    lanes::Int # number of lanes in roadway
    cars::Int # number of cars on roadway, including egovehicle
    dt::Float64 # timestep
    maxticks::Int # max ticks per episode
    rooms::Array{Float64, 2} # room between cars
    stadium::Bool # stadium roadway
    hri::Bool # HRI specific test case
    curriculum::Bool # (randomised) curriculum of cars and gaps during training
    carlength::Float64 # car length

    ego_pos::Int # location of egovehicle, between [1, cars]

    mode::String # types of other vehicles (mixed/cooperative/aggressive)
    video::Bool # save video
    write_data::Bool # save data file
end

"""
    dict_to_params(params::Dict)
convert simulation parameters dictionary to structure
"""
function dict_to_params(params::Dict)
    road = get(params, "length", 1000.0)
    lanes = get(params, "lanes", 3)
    cars = get(params, "cars", 50)
    dt = get(params, "dt", 0.2)
    maxticks = get(params, "maxticks", 200)
    stadium = get(params, "stadium", false)
    hri = get(params, "hri", true)
    curriculum = get(params, "curriculum", false)
    carlength = get(params, "carlength", 4.0)

    room = carlength * 1.1
    if curriculum
        cars = rand(10:cars)
        room = (room * rand() + 6.0)
    end

    if hri
        cars = max(50, cars)
    end

    cars_per_lane = Int(ceil(cars/lanes))
    room = stadium ? room / 2.0 : room
    rooms = zeros(lanes, cars_per_lane)
    for l in 1:lanes
        rooms[l, :] = cumsum(((0.6 * rand(cars_per_lane)) .+ 1) * room)
        # rooms[l, :] = cumsum(rand(cars_per_lane) .+ 6.0) # Sangjae's scheme
    end

    ego_pos = rand(1:cars)
    if hri
        if lanes == 2
            valid = collect(1:2:cars)
            num_valid = length(valid)
            ego_pos = rand(valid[max(num_valid-7, 1):end])
        elseif lanes == 3
            valid = collect(2:3:cars)
            num_valid = length(valid)
            ego_pos = rand(valid[max(num_valid-7, 1):end])
        end
    end

    mode = get(params, "eval_mode", "mixed")
    video = get(params, "video", false)
    write_data = get(params, "write_data", false)

    HRIParams(road, lanes, cars, dt, maxticks, rooms, stadium,
                hri, curriculum, carlength, ego_pos,
                mode, video, write_data)
end

"""
    get_vehicle(params::HRIParams, roadway::Roadway{Float64},
                                                pos::Int, id::Int)
create a vehicle to be put on the road
"""
function get_vehicle(params::HRIParams, roadway::Roadway{Float64},
                                                            pos::Int, id::Int)
    if params.stadium
        segment = (pos % 2) * 6 + (1 - pos % 2) * 3
        lane = params.lanes - ((floor((pos + 1) / 2) - 1) % params.lanes)
    else
        segment = 1
        lane = params.lanes - (pos % params.lanes)
    end
    segment = Int(segment)
    lane = Int(lane)

    lane0 = LaneTag(segment, lane)
    s0 = params.rooms[lane, Int(ceil(pos/params.lanes))]
    v0 = rand() + 1.0
    t0 = (rand() - 0.5) * (DEFAULT_LANE_WIDTH/2.0)
    ϕ0 = (2 * rand() - 1) * 0.1
    posF = Frenet(roadway[lane0], s0, t0, ϕ0)

    vehicle = Vehicle(VehicleState(posF, roadway, v0),
                                                VehicleDef(), id)

    vehicle
end

"""
    populate_scene(params::HRIParams, roadway::Roadway{Float64})
put all vehicles on the road
egovehicle is scene[EGO_ID]
deadend is scene[101]
"""
function populate_scene(params::HRIParams, roadway::Roadway{Float64})
    scene = Scene()
    carcolours = Dict{Int, Colorant}()
    models = Dict{Int, DriverModel}()

    ego = get_vehicle(params, roadway, params.ego_pos, EGO_ID)
    push!(scene, ego)
    carcolours[EGO_ID] = COLOR_CAR_EGO

    room = params.carlength * 1.1
    ego_lanetag = ego.state.posF.roadind.tag
    ego_s = params.rooms[ego_lanetag.lane,
                                        Int(ceil(params.ego_pos/params.lanes))]
    s_deadend = ego_s + (35.0 * rand() + 5.0)
    ignore_idx = findall(x -> x,
                abs.(params.rooms[ego_lanetag.lane, :] .- s_deadend) .< room)
    posF = Frenet(roadway[ego_lanetag], s_deadend, 0.0, 0.0)
    push!(scene, Vehicle(VehicleState(posF, roadway, 0.0),
                                                    VehicleDef(), 101))
    models[101] = ProportionalSpeedTracker()
    carcolours[101] = RGB(0, 0, 0)
    AutomotiveDrivingModels.set_desired_speed!(models[101], 0.0)

    ignored_cars = Dict(ego_lanetag.lane => ignore_idx)

    v_num = EGO_ID + 1
    for i in 1:(params.cars)
        if params.stadium
            segment = (i % 2) * 6 + (1 - i % 2) * 3
            lane = params.lanes - ((floor((i + 1) / 2) - 1) % params.lanes)
        else
            segment = 1
            lane = params.lanes - (i % params.lanes)
        end
        segment = Int(segment)
        lane = Int(lane)

        if lane ∈ keys(ignored_cars) &&
                Int(ceil(i/params.lanes)) ∈ ignored_cars[lane]
            continue
        end

        if i ≠ params.ego_pos
            vehicle = get_vehicle(params, roadway, i, v_num)
            push!(scene, vehicle)

            η_coop = nothing
            if params.mode == "cooperative"
                η_coop = 1.0
            elseif params.mode == "aggressive"
                η_coop = 0.0
            else
                η_coop = rand()
            end

            η_percept = (rand() - 0.5) * (0.15/0.5)
            stop_and_go = rand() > 0.75
            models[v_num] = BafflingDriver(params.dt,
                                    η_coop=η_coop,
                                    η_percept=η_percept,
                                    isStopAndGo=stop_and_go,
                                    stop_go_cycle=(rand()*5.0)+12.5,
                                    stop_period=(rand()*2.0)+7.0,
                                    mlon=BafflingLongitudinalTracker(
                                        δ=rand()+3.5,
                                        T=rand()+1.0,
                                        s_min=(rand()*2.0)+1.0,
                                        a_max=rand()+2.5,
                                        d_cmf=rand()+1.5,
                                        ΔT=params.dt,
                                        ),
                                    )
            carcolours[v_num] = HSV(0, 1.0 - η_coop, 1.0)

            v_des = (rand() * 3.0) + 2.0
            AutomotiveDrivingModels.set_desired_speed!(models[v_num], v_des)
            v_num += 1
        end
    end

    (scene, models, carcolours)
end

function make_scene(paramset::Dict=Dict())
    params = dict_to_params(paramset)

    if params.stadium
        roadway = gen_stadium_roadway(params.lanes, length=params.road,
                                                        width=0.0, radius=10.0)
    else
        roadway = gen_straight_roadway(params.lanes, params.road)
    end

    scene, models, colours = populate_scene(params, roadway)

    (roadway, scene, models, colours)
end

# # TEST
# using Blink
# using Interact
# using AutoViz
#
# # I had to write these lines because `using NNMPC` throws this error
# # '''
# # ERROR: LoadError: syntax: extra token "(" after end of expression
# # in expression starting at /home/dsaxena/.julia/packages/NNMPC/0pX7d/src/NNMPC.jl:21
# # '''
# using Distributions
# using Parameters
# include("baffling_drivers.jl")
#
# roadway, scene, models, colours = make_scene()
# w = Window()
# ui = @manipulate for frame in 1:1
#     render(scene, roadway, cam=CarFollowCamera(EGO_ID, 8.0), car_colors=colours)
# end
# body!(w, ui)

using DelimitedFiles
using Statistics
using AutoViz

"""
    collision_check(
            veh₁::Entity{VehicleState, D, Int},
            veh₂::Entity{VehicleState, D, Int}) where {D}
check for collision between two vehicles
"""
function collision_check(
            veh₁::Entity{VehicleState, D, Int},
            veh₂::Entity{VehicleState, D, Int}) where {D}
    x₁, y₁, θ₁ = veh₁.state.posG.x, veh₁.state.posG.y, veh₁.state.posG.θ
    x₂, y₂, θ₂ = veh₂.state.posG.x, veh₂.state.posG.y, veh₂.state.posG.θ
    l = (max(veh₁.def.length, veh₂.def.length) * 1.01) / 2.0
    w = (max(veh₁.def.width, veh₂.def.width) * 1.01) / 2.0

    r = w
    min_dist = Inf
    for i in [-1, 0, 1]
        for j in [-1, 0, 1]
            dist = sqrt(
                      ((x₁ + i *  (l - r) * cos(θ₁)) -
                                (x₂ + j * (l - r) * cos(θ₂)))^2
                     + ((y₁ + i * (l - r) * sin(θ₁)) -
                                (y₂ + j * (l - r) * sin(θ₂)))^2
                     )     - 2 * r
            min_dist = min(dist, min_dist)
        end
    end
    return max(min_dist, 0)
end

"""
    get_sim_data(rec::EntityQueueRecord{S,D,I}, roadway::Roadway{Float64},
                dt::Float64)
calculate the evaluation metrics from a list of scenes
"""
function get_sim_data(rec::EntityQueueRecord{S,D,I}, roadway::Roadway{Float64},
                dt::Float64)
    merge_tick = -1
    merge_count = 0
    min_dist = Inf
    ego_data = Dict{String, Vector{Float64}}(
                    "deviation" => [],
                    "vel" => [])

    ticks = nframes(rec)
    for frame_index in 1:ticks
        scene = rec[frame_index-ticks]
        ego = scene[findfirst(EGO_ID, scene)]

        ego_lanetag = ego.state.posF.roadind.tag
        target_lanetag = LaneTag(ego_lanetag.segment, ego_lanetag.lane+1)

        # track time to merge
        if merge_tick == -1
            if ego_lanetag.lane == target_lanetag.lane
                merge_tick = frame_index
            end
        end
        # # Dhruv's preferred method for determining a merge
        # if ego_lanetag.lane == target_lanetag.lane
        #     merge_count += 1
        #     if merge_tick == -1 && merge_count ≥ Int(10 / dt)
        #         merge_tick = frame_index
        #     end
        # else
        #     if merge_count > 0
        #         merge_count = 0
        #     end
        # end

        # track mininmum distance to other vehicles
        for veh in scene
            if veh.id != EGO_ID
                dist = collision_check(ego, veh)
                min_dist = min(dist, min_dist)
            end
        end

        # track desired lane offset
        ego_proj = Frenet(ego.state.posG,
                        roadway[target_lanetag], roadway)
        Δt = abs(ego_proj.t)
        push!(ego_data["deviation"], Δt)
        push!(ego_data["velocity"], ego.state.v)
    end

    (merge_tick, ticks, min_dist, ego_data)
end

"""
    write_data(env::EnvState, filename::String="default.dat")
log data for last episode
"""
function write_data(
                rec::EntityQueueRecord{S,D,I},
                roadway::Roadway{Float64},
                params::HRIParams,
                filename::String="default.dat") where {S<:VehicleState,D,I,R}

    merge_tick, ticks, min_dist, ego_data = get_sim_data(
                                                        rec, roadway, params.dt)
    offsets = ego_data["deviation"]
    vels = ego_data["velocity"]

    open(filename, "w") do f
        merge_tick = merge_tick
        ticks = ticks
        min_dist = min_dist
        avg_offset = mean(offsets)
        write(f, "$merge_tick\n")
        write(f, "$ticks\n")
        write(f, "$min_dist\n")
        write(f, "$avg_offset\n")

        val = reshape(offsets, (1, length(offsets)))
        write(f, "offsets,")
        writedlm(f, val, ",")

        val = reshape(vels, (1, length(vels)))
        write(f, "velocity,")
        writedlm(f, val, ",")
    end
