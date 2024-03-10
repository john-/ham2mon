# Example of json-server Channel Logger and Home Assistant

This is a example of sending json messages from Ham2mon to [Home Assistant](https://www.home-assistant.io) automation software.  Home assitant is seperate application and not related to Ham2mon.

## Dependencies
In addition to the standard ham2mon dependencies, the json-server channel logger requires the python `requests` library.  It is imported if this channel logger is used.

For this exmaple, Home Assistant needs to be running on the same network as Ham2mon.

## Command line arguments

These are the specific command line parameters required for this to function.  These are in addition to any others required for your situation.
```
--log_type json-server --log_target "<Home Assistant url>"
```
The Home Assistant (HA) url can be copy/pasted from Home Assistant|Settings|Automation & scenes|<your automation>|Webhook Trigger.  Even though HA shows only the Webhook ID the copy include the url (includes the ID).

## Home Assistant configuration

This example was ceated with the GUI automation editor and is included as YAML for easy reference.
```
alias: Notify when frequency has voice recording
description: ""
trigger:
  - platform: webhook
    allowed_methods:
      - POST
      - PUT
    local_only: true
    webhook_id: "<your ID"
condition:
  - condition: template
    value_template: >-
      {{ trigger.json['frequency'] == 460.175 and trigger.json['classification']
      == 'V' }}
  - condition: template
    value_template: >-
      {{(as_timestamp(now()) - as_timestamp(this.attributes.last_triggered)) |
      int(0) > 30 }}
action:
  - service: notify.persistent_notification
    data:
      message: >-
        Voice transmisson recorded - ({{ trigger.json['frequency'] }}/{{
        trigger.json['file'] }})
mode: single

```

## Important notes
Ham2mon can generate a lot of messages which could result in a significant amount of network traffic and overhead on Home Assistant.  Although the example automation has some throttling (triggers no more than once every 30 seconds) all the messages are still received by HA.

Additional ham2mon options can improve this.  For example, recording voice only and locking out channels/ranges that are not of interest.