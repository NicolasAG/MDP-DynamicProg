# MDP-DynamicProg

Policy Iteration, Value Iteration and Prioritized Sweeping for simple grid world MDP control.

## Grid World:

Rewards:

```
00  00 00 ... 00  00  00
00 -10 00 ... 00 -10  00
00  00 00 ... 00  00  00
          ...
00  00 00 ... 00  00  00
00  00 00 100 00  00  00
00  00 00 ... 00  00  00
          ...
00  00 00 ... 00  00  00
00 -10 00 ... 00 -10  00
00  00 00 ... 00  00  00
```

5 terminal states: 4 negative corners, 1 positive center

Actions:

```
Left(0) | Right(1) | Up(2) | Down(3)
```

## Usage:

`python main.py <algorithm> <OPTIONAL_FLAGS>`

`<OPTIONAL_FLAGS>` are:
 - --gamma (default 0.9, must be in (0.0, 1.0) )
 - --width (default 5, must be odd integer in (5, 101) )

`<algorithms>` are:
 - Policy Iteration algorithm: `python main.py policy_iteration <OPTIONAL_FLAGS>`
 - Value Iteration algorithm: `python main.py value_iteration <OPTIONAL_FLAGS>`
 - Prioritized Sweeping algorithm: `python main.py prioritized_sweeping <OPTIONAL_FLAGS>`
