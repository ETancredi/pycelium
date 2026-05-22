# control/runtime_mutator.py

class Mutation:
    """Defines a single parameter change scheduled at a given step."""
    def __init__(self, step: int, option: str, value):
        """
        Args:
            step: When to apply this mutation
            option: Name of the option in the Options object
            value: New value or a function(old_value) -> new_value
        """
        self.step = step          # Simulation step at which to trigger this mutation
        self.option = option      # Name of the attribute on the options object to change
        self.value = value        # Either a direct new value or a function to compute it

class RuntimeMutator:
    """Applies scheduled parameter mutations during simulation."""
    
    def __init__(self):
        self.mutations = []       # List to hold all scheduled Mutation instances

    def schedule(self, step: int, option: str, value):
        """Add a mutation to change 'option' at 'step'."""
        # Create and store a Mutation so it can be applied later
        self.mutations.append(Mutation(step, option, value))

    def apply(self, step: int, options):
        """Check and apply all mutations at this step."""
        for mut in self.mutations:
            if mut.step == step:  # If a mutation is scheduled for the current step…
                # Retrieve the current value of the option (or None if missing)
                current = getattr(options, mut.option, None)
                # Determine the new value: call mut.value if it's a function, else use it directly
                new_value = mut.value(current) if callable(mut.value) else mut.value
                # Set the new value on the options object
                setattr(options, mut.option, new_value)
                # Inform the user/developer about the change
                print(f"🔧 Mutation at step {step}: {mut.option} changed from {current} → {new_value}")
