function calc_fan(dims...; mode = "fan_in")
  @assert mode in ["fan_in", "fan_out"]
  num_dims = length(dims)
  num_dims < 2 && throw("Number of dimensions should be more than 1!")

  fan_out = dims[1]
  fan_in = dims[2]
  # if num_dims > 2
  #   receptive_field = prod(dims[1:end-2])
  #   fan_in  *= receptive_field
  #   fan_out *= receptive_field
  # end

  mode == "fan_in" ? fan_in : fan_out
end

function calc_gain(nonlinearity, a = nothing)
  linear_fns = ["linear", "conv1d", "conv2d", "conv3d", "conv_transpose1d", "conv_transpose2d", "conv_transpose3d"]
  if nonlinearity in linear_fns || nonlinearity == "sigmoid" || nonlinearity == "σ"
    return 1.0
  elseif nonlinearity == "tanh"
    return 5.0 / 3
  elseif nonlinearity == "relu"
    return √2.0
  elseif nonlinearity == "leakyrelu"
    if a == nothing
      negative_slope = 0.01
    elseif !(a isa Bool) && a isa Integer || a isa AbstractFloat
      # true/false are instances of Integer, hence check above
      negative_slope = a
    else
      throw("negative_slope $a not a valid number.")
    end
    return √(2.0 / (1 + negative_slope ^ 2))
  else
    throw("Unsupported nonlinearity $nonlinearity.")
  end
end

function kaiming_uniform(dims...; a = 0, mode = "fan_in", nonlinearity = "leakyrelu")
  fan = calc_fan(dims...; mode = mode)
  gain = calc_gain(nonlinearity, a)
  (rand(dims...) .- 0.5) .* √(12.0 / fan) .* gain
end
