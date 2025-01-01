// odin run particles.odin -file -debug -vet -strict-style -max-error-count=6 -define:RAYLIB_SHARED=true
package main

//import "core:math"
import "core:math/rand"
//import "core:strings"
import rl "vendor:raylib"
import rlgl "vendor:raylib/rlgl"
import "core:fmt"
import "core:math/linalg"
WORKGROUP_SIZE :: 1024
WINDOW_WIDTH :: 800
WINDOW_HEIGHT :: 800
Vector4 :: distinct [4]f32

get_random_float :: proc(from, to: f32) -> f32 {
    return from + (to - from) * rand.float32()
}

main :: proc() {
    rl.InitWindow(WINDOW_WIDTH, WINDOW_HEIGHT, "GPU Compute Shader Particles")
    defer rl.CloseWindow()

    fmt.println( "GL Version:", rlgl.GetVersion() )
    if rlgl.GetVersion() < rlgl.GlVersion.OPENGL_43 {
        //rlgl.GRAPHICS_API_OPENGL_43 = true
        fmt.println("Re-compile Raylib with GRAPHICS_API_OPENGL_43 = true, getting a new raylib.dll, then compile this with -define:RAYLIB_SHARED=true ")
    }
    
    // Compute shader for updating particles.
    shader_code := rl.LoadFileText("Shaders/particle_compute.glsl")
    shader_data := rlgl.CompileShader(cstring(shader_code), rlgl.COMPUTE_SHADER)
    compute_shader := rlgl.LoadComputeShaderProgram(shader_data)
    rl.UnloadFileText(shader_code)

    // Load render shader
    particle_shader := rl.LoadShader(
        "Shaders/particle_vertex.glsl",
        "Shaders/particle_fragment.glsl",
    )
    assert(rl.IsShaderValid(particle_shader))

    // Initialize particle data
    num_particles := WORKGROUP_SIZE * 10 //100
    positions := make([]Vector4, num_particles)
    velocities := make([]Vector4, num_particles)
    defer delete(positions)
    defer delete(velocities)

    // Initialize particles with random positions
    for i := 0; i < num_particles; i += 1 {
        positions[i] = Vector4{
            get_random_float(-0.5, 0.5),
            get_random_float(-0.5, 0.5),
            get_random_float(-0.5, 0.5),
            0,
        }
        velocities[i] = Vector4{0, 0, 0, 0}
    }
    assert( len(positions) == (num_particles) )

    // Load three buffers: Position, Velocity and Starting Position. Read/Write=RL_DYNAMIC_COPY.
    num_bytes := u32(len(positions) * size_of(Vector4))
    ssbo_positions := rlgl.LoadShaderBuffer(num_bytes, raw_data(positions), rlgl.DYNAMIC_COPY )
    ssbo_velocities := rlgl.LoadShaderBuffer( num_bytes, raw_data(velocities), rlgl.DYNAMIC_COPY )
    ssbo_start_positions := rlgl.LoadShaderBuffer(num_bytes, raw_data(positions), rlgl.DYNAMIC_COPY )
    fmt.println(ssbo_positions)
    fmt.println(ssbo_velocities)
    fmt.println(ssbo_start_positions)

    // For instancing we need a Vertex Array Object. 
    // Raylib Mesh* is inefficient for millions of particles.
    // For info see: https://www.khronos.org/opengl/wiki/Vertex_Specification
    particle_vao := rlgl.LoadVertexArray()
    rlgl.EnableVertexArray(particle_vao)

    // Define base triangle vertices
    vertices := [3]rl.Vector3{
        {-0.86, -0.5, 0.0},
        {0.86, -0.5, 0.0},
        {0.0, 1.0, 0.0},
    }

    vertices1 := [][]f32{
        {-0.86, -0.5, 0.0},
        {0.86, -0.5, 0.0},
        {0.0, 1.0, 0.0},
    }
    
    vertices2 :=[3][3]f32 {
        { -0.86, -0.5, 0.0 },
        { 0.86, -0.5, 0.0 },
        { 0.0,  1.0, 0.0 },
    }
    fmt.println( "vertices:[3]RL size_of", size_of(vertices) ) // 36 bytes
    fmt.println( "vertices:[][] size_of", size_of(vertices1) ) // 16 bytes
    fmt.println( "vertices:[3][3]f32 size_of", size_of(vertices2) ) // 36 bytes
    // Why does a 2D array have a different size_of when array size not explicitly declared?

    // Configure the vertex array with a single attribute of vec3.
    // This is the input to the vertex shader
    rlgl.EnableVertexAttribute(0)
    rlgl.LoadVertexBuffer( raw_data(vertices[:]) /*Pointer to data*/, size_of(vertices) /*Stride bytes*/ , false) // dynamic=false
    rlgl.SetVertexAttribute(index = 0, compSize = 3, type = rlgl.FLOAT, normalized = false, stride = 0, pointer = rawptr(nil),)
    rlgl.DisableVertexArray()

    // Camera setup
    camera := rl.Camera3D{
        position = {2, 2, 2},
        target = {0, 0, 0},
        up = {0, 1, 0},
        fovy = 35.0,
        projection = .PERSPECTIVE,
    }

    // Simulation parameters
    state := struct {
        time: f32,
        time_scale: f32,
        sigma: f32,
        rho: f32,
        beta: f32,
        particle_scale: f32,
        instances_x1000: f32,
    }{
        time = 0,
        time_scale = 0.2,
        sigma = 10,
        rho = 28,
        beta = 8.0/3.0,
        particle_scale = 1.0,
        instances_x1000 = 100.0,
    }

    eye := rl.Matrix(1)
    fmt.println(eye)
    fmt.println(compute_shader)

    numframes := 1

    for !rl.WindowShouldClose() {
        delta_time := rl.GetFrameTime()
        num_instances := int(state.instances_x1000 / 1000.0 * f32(num_particles))
        rl.UpdateCamera(&camera, .ORBITAL)

        // Compute Pass
        num_workgroups_to_launch := u32(num_particles/WORKGROUP_SIZE)
        {
            rlgl.EnableShader(compute_shader)

            // Set uniforms
            //proc(locIndex: c.int, value: rawptr, uniformType: c.int, count: c.int) ---
            rlgl.SetUniform(0, &state.time, i32(rl.ShaderUniformDataType.FLOAT), 1)
            rlgl.SetUniform(1, &state.time_scale, i32(rl.ShaderUniformDataType.FLOAT), 1)
            rlgl.SetUniform(2, &delta_time, i32(rl.ShaderUniformDataType.FLOAT), 1)
            rlgl.SetUniform(3, &state.sigma,i32(rl.ShaderUniformDataType.FLOAT), 1)
            rlgl.SetUniform(4, &state.rho, i32(rl.ShaderUniformDataType.FLOAT), 1)
            rlgl.SetUniform(5, &state.beta, i32(rl.ShaderUniformDataType.FLOAT), 1)

            rlgl.BindShaderBuffer(ssbo_positions, 0)
            rlgl.BindShaderBuffer(ssbo_velocities, 1)
            rlgl.BindShaderBuffer(ssbo_start_positions, 2)

            rlgl.ComputeShaderDispatch( num_workgroups_to_launch, 1, 1)
            rlgl.DisableShader()
        }
        
        
        rl.BeginDrawing()
        rl.ClearBackground(rl.BLACK)

                // GUI Pass
        {
            rl.GuiSlider(
                {550, 10, 200, 10},
                "Particles x1000",
                rl.TextFormat("%.2f", state.instances_x1000),
                &state.instances_x1000,
                0,
                1000,
            )
            rl.GuiSlider(
                {550, 25, 200, 10},
                "Particle Scale",
                rl.TextFormat("%.2f", state.particle_scale),
                &state.particle_scale,
                0,
                5,
            )
            rl.GuiSlider(
                {550, 40, 200, 10},
                "Speed",
                rl.TextFormat("%.2f", state.time_scale),
                &state.time_scale,
                0,
                1.0,
            )
            rl.GuiSlider(
                {650, 70, 100, 10},
                "Sigma",
                rl.TextFormat("%2.1f", state.sigma),
                &state.sigma,
                0,
                20,
            )
            rl.GuiSlider(
                {650, 85, 100, 10},
                "Rho",
                rl.TextFormat("%2.1f", state.rho),
                &state.rho,
                0,
                30,
            )
            rl.GuiSlider(
                {650, 100, 100, 10},
                "Beta",
                rl.TextFormat("%2.1f", state.beta),
                &state.beta,
                0,
                10,
            )

            state.time += delta_time

            if rl.GuiButton({350, 10, 100, 20}, "Restart (Space)") || rl.IsKeyPressed(.SPACE) {
                state.time = 0
            }
            if rl.GuiButton({280, 10, 60, 20}, "Reset") {
                state.time = 0
                state.time_scale = 0.2
                state.sigma = 10
                state.rho = 28
                state.beta = 8.0/3.0
                state.particle_scale = 1.0
                state.instances_x1000 = 100.0
            }

            rl.DrawFPS(10, 10)
            rl.DrawText(
                rl.TextFormat("N=%d", num_instances),
                10,
                30,
                20,
                rl.DARKGRAY,
            )
        }

        // Render Pass
        {
            rl.BeginMode3D(camera)
            rlgl.EnableShader(particle_shader.id)


            // Use custom camera matrices, or            
            // aspect := WINDOW_WIDTH / f32(WINDOW_HEIGHT)
            // my_projection := linalg.matrix4_perspective_f32(fovy=3.14/4.0, aspect=aspect, near=0.1, far=1000.0, flip_z_axis = true)
            // my_view := linalg.matrix4_look_at_f32(eye=[3]f32{2,2,2}, centre=[3]f32{0,0,0}, up=[3]f32{0,1,0}, flip_z_axis=true)
            // my_projection = linalg.transpose(my_projection)
            // my_view = linalg.transpose(my_view)
            
            // projection:rl.Matrix
            // view:rl.Matrix

            // for i in 0..<4 {
            //     for j in 0..<4 {
            //         projection[i][j] = my_projection[i][j]
            //         view[i][j] = my_view[i][j]
            //     }
            // }
            //rlgl.SetMatrixProjection(projection)
            //rlgl.SetMatrixModelview(view)
            // Or use the camera system in Raylib:
            projection := rlgl.GetMatrixProjection()
            view := rl.GetCameraMatrix(camera)

            // Caveman printf debugging
            // if numframes % 1000 == 0 {
            //   fmt.println( "num_workgroups_to_launch:", num_workgroups_to_launch, "num_instances:", num_instances, "particle_scale", state.particle_scale )
            // }

            rl.SetShaderValueMatrix(particle_shader, 0, projection)
            rl.SetShaderValueMatrix(particle_shader, 1, view)
            rl.SetShaderValue(particle_shader, 2, &state.particle_scale, rl.ShaderUniformDataType.FLOAT)

            rlgl.BindShaderBuffer(ssbo_positions, 0)
            rlgl.BindShaderBuffer(ssbo_velocities, 1)

            rlgl.EnableVertexArray(particle_vao)
            rlgl.DrawVertexArrayInstanced(0, 3, i32(num_instances) )
            rlgl.DisableVertexArray()
            rlgl.DisableShader()

            rl.DrawCubeWires({0, 0, 0}, 1.0, 1.0, 1.0, rl.DARKGRAY)
            rl.EndMode3D()
        }


        rl.EndDrawing()
        numframes += 1
    }
}