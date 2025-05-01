class Sender:
    """
    Represents the sender (Alice) in the identification system.
    Encodes messages using a provided encoder.
    """
    def __init__(self, encoder):
        """
        Initialize the sender with an encoder.
        
        Args:
            encoder: An instance of Encoder that will be used to encode messages
        """
        self.encoder = encoder
    
    def send(self, message):
        """
        Encode and send a message.
        
        Args:
            message: The message to encode and send
            
        Returns:
            The encoded message
        """
        return self.encoder.encode(message)


class Receiver:
    """
    Represents the receiver (Bob) in the identification system.
    Uses the same encoder as the sender to determine if the
    received encoded message matches a candidate message.
    """
    def __init__(self, encoder):
        """
        Initialize the receiver with an encoder.
        
        Args:
            encoder: An instance of Encoder that will be used to encode and compare messages
        """
        self.encoder = encoder
    
    def identify(self, encoded_message, candidate_message):
        """
        Identify whether the candidate message matches the received encoded message.
        
        Args:
            encoded_message: The encoded message received from the sender
            candidate_message: The candidate message to check against
            
        Returns:
            bool: True if the encoded candidate message matches the received encoded message
        """
        return self.encoder.encode(candidate_message) == encoded_message


class IdentificationSystem:
    """
    Represents the complete identification system, combining a sender and receiver.
    """
    def __init__(self, encoder):
        """
        Initialize the identification system with an encoder.
        
        Args:
            encoder: An instance of Encoder to be used by both sender and receiver
        """
        self.encoder = encoder
        self.sender = Sender(encoder)
        self.receiver = Receiver(encoder)
    
    def run_identification(self, sender_message, receiver_message):
        """
        Run the complete identification process.
        
        Args:
            sender_message: The message from the sender (Alice)
            receiver_message: The message the receiver (Bob) wants to check
            
        Returns:
            tuple: (encoded_message, is_match)
                encoded_message: The encoded message sent by the sender
                is_match: Boolean indicating whether the receiver's message matches
        """
        encoded_message = self.sender.send(sender_message)
        is_match = self.receiver.identify(encoded_message, receiver_message)
        return encoded_message, is_match