import styled, { css } from 'styled-components';
export const Grid = styled.div`
  display: grid;
  grid-column-gap: ${({ spacious, tiny, noGap }) =>
    noGap ? 0 : spacious ? '5rem' : tiny ? '.75rem' : '1.875rem'};
  grid-row-gap:  ${({ spacious, tiny, noGap }) =>
    noGap ? 0 : spacious ? '5rem' : tiny ? '.75rem' : '1.875rem'};
  ${({ noMargin }) => !noMargin && 'margin-bottom: 1.5rem;'}
  text-align: left;
  grid-auto-flow: ${({ autoFlow = 'row' }) => autoFlow};

  ${({ centered }) =>
    centered &&
    css`
      justify-content: center;
    `}

  ${({ verticalAlign }) =>
    verticalAlign &&
    css`
      align-items: ${verticalAlign};
    `}
  
  @media (min-width: 768px) {
    justify-content: ${({ justify = 'space-between' }) => justify};
    grid-template-columns: ${({ columns = '1fr 1fr' }) => columns};
  }
`;
Grid.displayName = 'Grid';
