import React from 'react';
import tinycolor from 'tinycolor2';
import {phOuter, phInner, phText} from './styles.module.css';

function Ph({children, bg, h, aspect}) {
	return (<div className={phOuter} style={{
		color: `${bg && tinycolor(bg).isDark() ? 'white' : 'black'}`,
  	backgroundColor: `${bg || 'rgba(0,0,0,0.1)'}`,
  	[h ? 'height' : 'paddingBottom']: `${h ? `${h}px` : `${100 / (aspect || 1)}%`}`

	}}>
		<div className={phInner}>
			<div className={phText}>{children}</div>
		</div>
	</div>)
}

export default Ph;
