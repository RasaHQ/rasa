import React from 'react';

const Button = (props) => (
  <button
    {...props}
    style={{
      backgroundColor: '#5a17ee',
      border: '1px solid transparent',
      color: '#fff',
      borderRadius: 8,
      padding: 12,
      fontSize: 15,
      fontWeight: '600',
      cursor: 'pointer',
      ...props.style,
    }}
  />
);

export default Button;
