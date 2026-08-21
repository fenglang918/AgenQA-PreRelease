# Thermal Transport Through a Two-Layer Composite Rod

## A synthetic scientific paper for the AgenQA public demo

This paper was authored specifically as a distributable test input. It does not
report a real experiment and does not reproduce text or data from another work.

## Abstract

We study steady one-dimensional heat conduction through a composite rod made of
two homogeneous layers in perfect thermal contact. A closed-form resistance
model gives the heat flux, interface temperature, and sensitivity to layer
thickness. A small parameter study illustrates why the lower-conductivity layer
dominates the total temperature drop. The model is intentionally compact but
contains enough linked facts to support multi-step scientific reasoning tasks.

## 1. Physical model

The rod has cross-sectional area A and two layers arranged in series. Layer 1
has length L1 and thermal conductivity k1. Layer 2 has length L2 and thermal
conductivity k2. The left boundary is held at temperature Th and the right
boundary at Tc, with Th > Tc. Lateral heat loss and contact resistance are
neglected, and all material properties are constant.

For steady one-dimensional conduction, each layer has thermal resistance

R1 = L1 / (k1 A),    R2 = L2 / (k2 A).

The heat rate is

Q = (Th - Tc) / (R1 + R2),

and the heat flux is q = Q / A. Because the same heat rate crosses both layers,
the interface temperature Ti can be written in either equivalent form:

Ti = Th - Q R1 = Tc + Q R2.

These equations also imply that the fraction of the total temperature drop
occurring in layer 2 is R2 / (R1 + R2).

## 2. Reference configuration

The reference rod uses A = 1.00e-2 m^2, L1 = 2.00e-2 m, k1 = 20 W m^-1 K^-1,
L2 = 3.00e-2 m, and k2 = 5 W m^-1 K^-1. Boundary temperatures are Th = 400 K
and Tc = 300 K.

For this configuration, R1 = 0.10 K W^-1 and R2 = 0.60 K W^-1. Therefore the
total resistance is 0.70 K W^-1, the heat rate is 142.857 W, and the heat flux
is 1.42857e4 W m^-2. The interface temperature is 385.714 K. Layer 2 accounts
for 6/7 of the total temperature drop, even though it contains only 3/5 of the
total rod length.

## 3. Thickness perturbation

We next double the thickness of layer 2 while holding A, k1, k2, Th, and Tc
fixed. The new layer-2 resistance is R2' = 1.20 K W^-1 and the total resistance
is 1.30 K W^-1. The new heat rate is Q' = 76.923 W. The ratio Q'/Q is 7/13,
showing that doubling one layer's thickness does not simply halve the heat rate
when another series resistance remains present.

The perturbed interface temperature is Ti' = Th - Q' R1 = 392.308 K. The
interface becomes hotter because the enlarged downstream resistance causes a
larger fraction of the total temperature drop to occur in layer 2.

## 4. Discussion

The series-resistance formulation exposes a short dependency chain. Geometry
and conductivity determine R1 and R2; the resistances determine Q; and Q with
either layer resistance determines Ti. A solver that is shown only the original
parameters and asked for the perturbed interface temperature must reconstruct
all intermediate quantities. Locally, however, every transition can be checked
using a single equation. This separation between local verification and global
reconstruction makes the example suitable for demonstrating Edge and Path
views.

## 5. Limitations

The model assumes constant conductivity, perfect thermal contact, and no lateral
losses. It should not be used to predict a real composite without checking those
assumptions. All values in this paper are synthetic and included only for
software demonstration.
