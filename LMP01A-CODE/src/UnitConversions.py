class UnitConversions:

    def A_to_uA(self, A: float) -> float:
        """Converts the current from A to uA.

        Args:
            A (float): The current in A.

        Returns:
            float: The current in uA.
        """
        return A * 1_000_000.0

    def uA_to_A(self, uA: float) -> float:
        """Converts the current from uA to A.

        Args:
            uA (float): The current in uA.

        Returns:
            float: The current in A.
        """
        return uA / 1_000_000.0

    def A_to_mA(self, A: float) -> float:
        """Converts the current from A to mA.

        Args:
            A (float): The current in A.

        Returns:
            float: The current in mA.
        """
        return A * 1_000.0

    def us_to_ms(self, us: float) -> float:
        """Converts the time from us to ms.

        Args:
            us (float): The time in us.

        Returns:
            float: The time in ms.
        """
        return us / 1000.0

    def us_to_s(self, us: float) -> float:
        """Converts the time from us to s.

        Args:
            us (float): The time in us.

        Returns:
            float: The time in s.
        """
        return us / 1_000_000.0

    def s_to_us(self, s: float) -> float:
        """Converts the time from s to us.

        Args:
            s (float): The time in s.

        Returns:
            float: The time in us.
        """
        return s * 1_000_000.0

    def s_to_ms(self, s: float) -> float:
        """Converts the time from s to ms.

        Args:
            s (float): The time in s.

        Returns:
            float: The time in ms.
        """
        return s * 1000.0

    def s_to_h(self, s: float) -> float:
        """Converts the time from s to h.

        Args:
            s (float): The time in s.

        Returns:
            float: The time in h.
        """
        return s / 3600.0

    def us_to_h(self, us: float) -> float:
        """Converts the time from us to h.

        Args:
            us (float): The time in us.

        Returns:
            float: The time in h.
        """
        return us / 3600_000_000.0
