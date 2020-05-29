if __name__ == '__main__':
  nh = 4
  use_interior = False

  xa = -.5
  xb = .5
  ya = -.5
  yb = .5
  dx = (xb - xa) / nh
  dy = (yb - ya) / nh
  print('nh = {}'.format(nh))
  print('x range [{:e}, {:e}], dx = {:e}'.format(xa, xb, dx))
  print('y range [{:e}, {:e}], dy = {:e}'.format(ya, yb, dy))

  total_segments = nh * 4
  num_interior = (nh-1) * (nh-1) if use_interior else 0
  num_exterior = nh * 4 # including corner points
  total_vertices = num_interior + num_exterior

  fn_poly = 'unit.poly'
  with open('/net/ohm/export/iss/inputs/meshes/non-obtuse/'+fn_poly, 'w') as f_poly:
    '''
    # First line: <# of vertices> <dimension (must be 2)> <# of attributes> <# of boundary markers (0 or 1)>
    # <# of vertices> may be set to zero to indicate that the vertices are listed in a separate .node file
    f_poly.write('{0} 2 0 1\n'.format(total_vertices))
    # Following lines: <vertex #> <x> <y> [attributes] [boundary marker]
    for n in range(0, total_vertices):
      if (n < num_interior):
        i = n // (nh-1)
        j = n % (nh-1)
        f_poly.write('{0}\t{1:e}\t{2:e}\t{3}\n'.format(n, xa+dx*(j+1), ya+dy*(i+1), 0))
      # top (from corner)
      elif (n < num_interior + nh):
        j = n - num_interior
        f_poly.write('{0}\t{1:e}\t{2:e}\t{3}\n'.format(n, xa+dx*j, yb, 1))
      # right (from corner)
      elif (n < num_interior + nh*2):
        i = n - (num_interior + nh)
        f_poly.write('{0}\t{1:e}\t{2:e}\t{3}\n'.format(n, xb, yb-dy*i, 1))
      # bottom (from corner)
      elif (n < num_interior + nh*3):
        j = n - (num_interior + nh*2)
        f_poly.write('{0}\t{1:e}\t{2:e}\t{3}\n'.format(n, xb-dx*j, ya, 1))
      # left (from corner)
      else:
        i = n - (num_interior + nh*3)
        f_poly.write('{0}\t{1:e}\t{2:e}\t{3}\n'.format(n, xa, ya+dy*i, 1))
    '''
    f_poly.write('{0} 2 0 1\n'.format(num_exterior))
    for n in range(0, num_exterior):
      # top (from corner)
      if (n < nh):
        f_poly.write('{0}\t{1:e}\t{2:e}\t{3}\n'.format(n, xa+dx*n, yb, 1))
      # right (from corner)
      elif (n < nh*2):
        i = n - nh
        f_poly.write('{0}\t{1:e}\t{2:e}\t{3}\n'.format(n, xb, yb-dy*i, 1))
      # bottom (from corner)
      elif (n < nh*3):
        j = n - nh*2
        f_poly.write('{0}\t{1:e}\t{2:e}\t{3}\n'.format(n, xb-dx*j, ya, 1))
      # left (from corner)
      else:
        i = n - nh*3
        f_poly.write('{0}\t{1:e}\t{2:e}\t{3}\n'.format(n, xa, ya+dy*i, 1))

    # One line: <# of segments> <# of boundary markers (0 or 1)>
    f_poly.write('{0} {1}\n'.format(total_segments, 1))
    # Following lines: <segment #> <endpoint> <endpoint> [boundary marker]
    for n in range(0, total_segments):
      if (n < total_segments-1):
        f_poly.write('{0}\t{1}\t{2}\t{3}\n'.format(n, num_interior+n, num_interior+n+1, 1))
      else:
        f_poly.write('{0}\t{1}\t{2}\t{3}\n'.format(n, num_interior+n, num_interior, 1))
    
    # One line: <# of holes>
    f_poly.write('0\n')
    # Following lines: <hole #> <x> <y>

    # Optional line: <# of regional attributes and/or area constraints>
    # Optional following lines: <region #> <x> <y> <attribute> <maximum area>
