# MDP-DynamicProg

Dynamic Programs for simple grid world MDP control.

## Grid World:

Rewards:

```
00  00  00  00  00
00 -10  00 -10  00
00  00 100  00  00
00 -10  00 -10  00
00  00  00  00  00
```

Actions:

```
Left(0) | Right(1) | Up(2) | Down(3)
```

## Usage:

Policy Iteration algorithm:
`python main.py policy_iteration`

Value Iteration algorithm:
`python main.py value_iteration`

Prioritize Sweeping algorithm:
`python main.py prioritize_sweeping`
