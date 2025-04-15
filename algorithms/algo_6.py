# algorithms/algo_6.py
# Author: @ledondodo

from algorithms.wsm import WSM


def simple_graph():
    """
    Create a simple graph with 3 states and 2 transitions

    Returns:
        wsm (WSM): word state machine
    """
    wsm = WSM()
    start_state = wsm.add_state(name="Start")
    state_one = wsm.add_state(name="One")
    state_two = wsm.add_state(name="Two")
    state_three = wsm.add_state(name="Three")
    end_state = wsm.add_state(name="End")
    wsm.set_start(start_state)
    wsm.set_final(end_state)
    wsm.add_symbol("a", 1)
    wsm.add_arc(start_state, state_one, 1)
    wsm.add_symbol("b", 2)
    wsm.add_arc(state_one, state_two, 2)
    wsm.add_symbol("c", 3)
    wsm.add_arc(state_two, end_state, 3)
    wsm.add_symbol("d", 4)
    wsm.add_arc(start_state, state_three, 4)
    wsm.add_symbol("e", 5)
    wsm.add_arc(state_three, end_state, 5)
    wsm.add_symbol("x", 10)
    wsm.add_arc(end_state, start_state, 10)
    wsm.compile()
    return wsm
