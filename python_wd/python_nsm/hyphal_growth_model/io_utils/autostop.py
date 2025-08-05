# io_utils/autostop.py

class AutoStop:
    """
    Monitors sim and stops it when no active tips remain.
    Ensures sim ends gracefully instead of running indefinitely.
    """
    
    def __init__(self, enabled=True, print_reason=True):
        """
        Initialise the AutoStop controller.
        Args:
            enabled (bool): If False, Autostop.check() will always return False (disabled).
            print_reason (bool): If True, prints a message explaining why the simulation stopped.
        """
        # Flag to turn automatic stopping on or off
        self.enabled = enabled
        # Flag to control printing of stop reasoning
        self.print_reason = print_reason

    def check(self, mycel, step):
        """
        Check whether the sim should stop at this step.
        Args:
            mycel: Mycel instance containing current simulation state.
            step (int): Current simulation step number.
        Returns:
            bool: True if simulation should stop, False otherwise.
        """
        # If AutoStop feature is disabled, never stop automatically
        if not self.enabled:
            return False
        
        # Retrieve list of active (alive) tips from the mycelium
        active_tips = mycel.get_tips()
        # If there are no active tips left, trigger a stop
        if len(active_tips) == 0:
            # Optionally print a message explaining why the sim has stopped
            if self.print_reason:
                print(f"🛑 AutoStop triggered: no active tips at step {step}")
            return True # Signal to caller that sim should end
        
        # Otherwise, tips remain: do not stop
        return False
