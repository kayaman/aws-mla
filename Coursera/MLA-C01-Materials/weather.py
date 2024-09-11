
import json

def lambda_handler(event, context):
    agent = event['agent']
    actionGroup = event['actionGroup']
    function = event['function']
    parameters = event.get('parameters', [])
    
    # Initialize variables for city and units
    city = None
    units = 'Fahrenheit'

    # Extract 'city' and 'units' from parameters
    for param in parameters:
        if param['name'] == 'city':
            city = param['value']
        elif param['name'] == 'units':
            units = param['value']

    # Log extracted parameters
    print(f"City: {city}, Units: {units}")

    # Execute your business logic here. For more information, refer to: https://docs.aws.amazon.com/bedrock/latest/userguide/agents-lambda.html
    responseBody =  {
        "TEXT": {
            "body": f"The weather in {city} is 75 degrees {units} and sunny!"
        }
    }

    action_response = {
        'actionGroup': actionGroup,
        'function': function,
        'functionResponse': {
            'responseBody': responseBody
        }

    }

    dummy_function_response = {'response': action_response, 'messageVersion': event['messageVersion']}
    print("Response: {}".format(dummy_function_response))

    return dummy_function_response
