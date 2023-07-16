# SGS models 

## Eddy viscosity models 

$$ \nabla . \tau = \nabla.(\nu_e \nabla \omega )$$


### Eddy viscosity, Smagorinsky
### Eddy viscosity, Leith

### Eddy viscosity, Dnamic Smagorinsky
### Eddy viscosity, Dynamic Leith

## Local models 


$$\Pi = ∇.(ν_e ∇ω )$$

versus 

$$\Pi= ν_e ∇.( ∇ω ) == ν_e (∇^2ω )$$


![pi](../assets/7940296/bd635cb3-ca3e-497c-9eb8-d032079cdb37)

Options 

1. Domain Averaged $C_s$
$$
\nu_e(x,t) = ( C_s(x,t) \Delta )^2 [ 2 |\bar{S}(x,t) |^2  ]^{(1/2)}
$$

2. Local $C_s$

$$
\nu_e(x,t) = ( C_s(x,t) \Delta )^2 [ 2 |\bar{S}(x,t) |^2  ]^{(1/2)}
$$

3. Domain average 


From Pierre Sagaut [1]
<img src="Assets/icon.png" width="50">

![sagaut](../assets/7940296/a488d7ed-c320-46b7-a2c5-c78bd37bcdde)


[1]
